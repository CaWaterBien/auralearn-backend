<?php

namespace App\Http\Controllers;

use App\Models\User;
use App\Models\UserProgress;
use App\Models\ActivitySubmission;
use App\Models\ActivityCertificate;
use App\Models\Course;
use App\Models\Lesson;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class DashboardController extends Controller
{
    /**
     * Get user dashboard statistics
     */
    public function getDashboardStats(Request $request)
    {
        try {
            $userId = $request->query('user_id', 1); // Default to user ID 1
            
            $user = User::find($userId);
            if (!$user) {
                return response()->json([
                    'message' => 'User not found'
                ], 404);
            }

            // Get total points from activity submissions (using raw SQL for PostgreSQL compatibility)
            $totalPoints = ActivitySubmission::where('user_id', $userId)
                ->whereRaw('is_completed IS TRUE')
                ->sum('score') ?? 0;

            // Calculate streak (simplified - days with activity)
            $recentActivity = ActivitySubmission::where('user_id', $userId)
                ->whereRaw('is_completed IS TRUE')
                ->orderBy('completed_at', 'desc')
                ->pluck('completed_at')
                ->map(function($date) {
                    return \Carbon\Carbon::parse($date)->startOfDay();
                })
                ->unique()
                ->values();

            $streak = $this->calculateStreak($recentActivity);

            // Get courses completed
            $totalCourses = Course::where('is_published', 1)->count();
            $completedCourses = UserProgress::where('user_id', $userId)
                ->where('is_completed', 1)
                ->whereNotNull('course_id')
                ->distinct('course_id')
                ->count('course_id');

            // Get global rank (based on total points)
            $rank = DB::table('users')
                ->join('activity_submissions', 'users.id', '=', 'activity_submissions.user_id')
                ->select('users.id', DB::raw('SUM(activity_submissions.score) as total_score'))
                ->whereRaw('activity_submissions.is_completed IS TRUE')
                ->groupBy('users.id')
                ->havingRaw('SUM(activity_submissions.score) > ?', [$totalPoints])
                ->count() + 1;

            // Get recent lessons
            $recentLessons = UserProgress::where('user_id', $userId)
                ->whereNotNull('lesson_id')
                ->with(['lesson', 'course'])
                ->orderBy('updated_at', 'desc')
                ->take(5)
                ->get()
                ->map(function($progress) {
                    return [
                        'title' => $progress->lesson->title ?? 'Unknown Lesson',
                        'progress' => round($progress->completion_percentage, 0),
                        'lastAccessed' => $progress->updated_at->toISOString(),
                        'courseTitle' => $progress->course->title ?? 'Unknown Course'
                    ];
                });

            // Get current course (most recent incomplete course)
            $currentCourse = UserProgress::where('user_id', $userId)
                ->where('is_completed', 0)
                ->whereNotNull('course_id')
                ->with('course')
                ->orderBy('updated_at', 'desc')
                ->first();

            // Get time spent learning (in minutes from activity submissions)
            $timeSpentMinutes = ActivitySubmission::where('user_id', $userId)
                ->sum('time_spent_minutes');
            
            $hours = floor($timeSpentMinutes / 60);
            $minutes = $timeSpentMinutes % 60;
            $timeSpent = "{$hours}h {$minutes}m";

            // Get lessons completed count
            $lessonsCompleted = UserProgress::where('user_id', $userId)
                ->where('is_completed', 1)
                ->whereNotNull('lesson_id')
                ->count();

            // Get average score
            $averageScore = ActivitySubmission::where('user_id', $userId)
                ->whereRaw('is_completed IS TRUE')
                ->avg('score');
            $averageScore = $averageScore ? round($averageScore, 0) : 0;

            // Get projects completed (count of completed activities)
            $projectsCompleted = ActivitySubmission::where('user_id', $userId)
                ->whereRaw('is_completed IS TRUE')
                ->distinct('activity_id')
                ->count('activity_id');

            // Get badges (from certificates)
            $badges = ActivityCertificate::where('user_id', $userId)
                ->orderBy('earned_at', 'desc')
                ->get()
                ->map(function($cert) {
                    $types = [
                        'completion' => 'Completion Master',
                        'excellence' => 'Excellence Award',
                        'first_attempt' => 'First Try Champion',
                        'perfect_score' => 'Perfect Score Hero'
                    ];
                    $levels = [
                        'bronze' => ' 🥉',
                        'silver' => ' 🥈',
                        'gold' => ' 🥇',
                        'platinum' => ' 💎'
                    ];
                    return ($types[$cert->certificate_type] ?? 'Achievement') . 
                           ($levels[$cert->badge_level] ?? '');
                })
                ->toArray();

            // Add welcome badge if no badges yet
            if (empty($badges)) {
                $badges = ['Welcome Badge 👋'];
            }

            // Calculate points gained this week
            $pointsThisWeek = ActivitySubmission::where('user_id', $userId)
                ->whereRaw('is_completed IS TRUE')
                ->where('completed_at', '>=', \Carbon\Carbon::now()->startOfWeek())
                ->sum('score');

            return response()->json([
                'totalPoints' => (int)$totalPoints,
                'pointsThisWeek' => (int)$pointsThisWeek,
                'streak' => $streak,
                'completedCourses' => $completedCourses,
                'totalCourses' => $totalCourses,
                'rank' => $rank,
                'currentCourse' => $currentCourse ? $currentCourse->course->title : null,
                'recentLessons' => $recentLessons,
                'timeSpent' => $timeSpent,
                'lessonsCompleted' => $lessonsCompleted,
                'averageScore' => $averageScore,
                'projectsCompleted' => $projectsCompleted,
                'badges' => $badges
            ]);

        } catch (\Exception $e) {
            \Log::error('Failed to fetch dashboard stats: ' . $e->getMessage());
            return response()->json([
                'message' => 'Failed to fetch dashboard statistics',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Calculate consecutive day streak
     */
    private function calculateStreak($dates)
    {
        if ($dates->isEmpty()) {
            return 0;
        }

        $streak = 1;
        $today = \Carbon\Carbon::now()->startOfDay();
        
        // Check if the most recent activity was today or yesterday
        $mostRecent = $dates->first();
        $daysSinceLastActivity = $today->diffInDays($mostRecent);
        
        if ($daysSinceLastActivity > 1) {
            return 0; // Streak broken
        }

        // Count consecutive days
        for ($i = 0; $i < $dates->count() - 1; $i++) {
            $current = $dates[$i];
            $next = $dates[$i + 1];
            
            $daysDiff = $current->diffInDays($next);
            
            if ($daysDiff === 1) {
                $streak++;
            } else {
                break; // Streak broken
            }
        }

        return $streak;
    }
}

