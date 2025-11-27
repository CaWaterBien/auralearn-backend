<?php

namespace App\Http\Controllers;

use App\Models\ActivityCertificate;
use App\Models\UserProgress;
use App\Models\Activity;
use App\Models\Course;
use App\Models\Lesson;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class AchievementController extends Controller
{
    /**
     * Get all achievements for a user
     */
    public function index(Request $request)
    {
        try {
            $userId = $request->query('user_id', 1); // Default to user ID 1 for development
            
            $achievements = ActivityCertificate::where('user_id', $userId)
                ->with(['activity', 'activity.lesson'])
                ->orderBy('earned_at', 'desc')
                ->get()
                ->map(function ($cert) {
                    $achievementData = is_string($cert->achievement_data) 
                        ? json_decode($cert->achievement_data, true) 
                        : $cert->achievement_data;
                    
                    return [
                        'id' => $cert->id,
                        'user_id' => $cert->user_id,
                        'activity_id' => $cert->activity_id,
                        'submission_id' => $cert->submission_id,
                        'certificate_id' => $cert->certificate_id,
                        'certificate_type' => $cert->certificate_type,
                        'badge_level' => $cert->badge_level,
                        'achievement_data' => array_merge($achievementData ?? [], [
                            'activity_title' => $cert->activity->title ?? 'Unknown Activity',
                            'lesson_title' => $cert->activity->lesson->title ?? 'Unknown Lesson',
                        ]),
                        'certificate_url' => $cert->certificate_url,
                        'is_verified' => $cert->is_verified,
                        'earned_at' => $cert->earned_at,
                        'created_at' => $cert->created_at,
                        'updated_at' => $cert->updated_at,
                    ];
                });

            return response()->json([
                'achievements' => $achievements
            ]);
        } catch (\Exception $e) {
            \Log::error('Failed to fetch achievements: ' . $e->getMessage());
            return response()->json([
                'message' => 'Failed to fetch achievements',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Get achievement statistics for a user
     */
    public function stats(Request $request)
    {
        try {
            $userId = $request->query('user_id', 1);
            
            $certificates = ActivityCertificate::where('user_id', $userId)->get();
            
            $stats = [
                'total_achievements' => $certificates->count(),
                'certificates_earned' => $certificates->count(),
                'total_points' => $certificates->sum(function($cert) {
                    $data = is_string($cert->achievement_data) 
                        ? json_decode($cert->achievement_data, true) 
                        : $cert->achievement_data;
                    return $data['score'] ?? 0;
                }),
                'badges' => [
                    'bronze' => $certificates->where('badge_level', 'bronze')->count(),
                    'silver' => $certificates->where('badge_level', 'silver')->count(),
                    'gold' => $certificates->where('badge_level', 'gold')->count(),
                    'platinum' => $certificates->where('badge_level', 'platinum')->count(),
                ],
                'recent_achievements' => $certificates->sortByDesc('earned_at')
                    ->take(5)
                    ->values()
                    ->map(function ($cert) {
                        $achievementData = is_string($cert->achievement_data) 
                            ? json_decode($cert->achievement_data, true) 
                            : $cert->achievement_data;
                        
                        return [
                            'id' => $cert->id,
                            'user_id' => $cert->user_id,
                            'activity_id' => $cert->activity_id,
                            'submission_id' => $cert->submission_id,
                            'certificate_id' => $cert->certificate_id,
                            'certificate_type' => $cert->certificate_type,
                            'badge_level' => $cert->badge_level,
                            'achievement_data' => array_merge($achievementData ?? [], [
                                'activity_title' => $cert->activity->title ?? 'Unknown Activity',
                            ]),
                            'is_verified' => $cert->is_verified,
                            'earned_at' => $cert->earned_at,
                        ];
                    })
            ];

            return response()->json($stats);
        } catch (\Exception $e) {
            \Log::error('Failed to fetch achievement stats: ' . $e->getMessage());
            return response()->json([
                'message' => 'Failed to fetch achievement statistics',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Get achievements grouped by course
     */
    public function byCourse(Request $request)
    {
        try {
            $userId = $request->query('user_id', 1);
            
            $courses = Course::where('is_published', 1)
                ->with(['lessons'])
                ->get()
                ->map(function ($course) use ($userId) {
                    // Get all activities for this course
                    $activityIds = Activity::whereHas('lesson', function($query) use ($course) {
                        $query->where('course_id', $course->id);
                    })->pluck('id');
                    
                    // Get achievements for these activities
                    $achievements = ActivityCertificate::where('user_id', $userId)
                        ->whereIn('activity_id', $activityIds)
                        ->with(['activity'])
                        ->get()
                        ->map(function ($cert) {
                            $achievementData = is_string($cert->achievement_data) 
                                ? json_decode($cert->achievement_data, true) 
                                : $cert->achievement_data;
                            
                            return [
                                'id' => $cert->id,
                                'user_id' => $cert->user_id,
                                'activity_id' => $cert->activity_id,
                                'submission_id' => $cert->submission_id,
                                'certificate_id' => $cert->certificate_id,
                                'certificate_type' => $cert->certificate_type,
                                'badge_level' => $cert->badge_level,
                                'achievement_data' => array_merge($achievementData ?? [], [
                                    'activity_title' => $cert->activity->title ?? 'Unknown Activity',
                                ]),
                                'certificate_url' => $cert->certificate_url,
                                'is_verified' => $cert->is_verified,
                                'earned_at' => $cert->earned_at,
                                'created_at' => $cert->created_at,
                                'updated_at' => $cert->updated_at,
                            ];
                        });
                    
                    // Get user progress for this course
                    $progress = UserProgress::where('user_id', $userId)
                        ->where('course_id', $course->id)
                        ->get();
                    
                    $totalLessons = $course->lessons->count();
                    $completedLessons = $progress->where('is_completed', 1)->count();
                    
                    $avgProgress = $progress->avg('completion_percentage') ?? 0;
                    
                    return [
                        'course_id' => $course->id,
                        'course_title' => $course->title,
                        'course_slug' => $course->slug,
                        'total_achievements' => $achievements->count(),
                        'certificates_earned' => $achievements->count(),
                        'progress_percentage' => round($avgProgress, 2),
                        'achievements' => $achievements->values(),
                        'lessons_completed' => $completedLessons,
                        'total_lessons' => $totalLessons,
                        'badges' => [
                            'bronze' => $achievements->where('badge_level', 'bronze')->count(),
                            'silver' => $achievements->where('badge_level', 'silver')->count(),
                            'gold' => $achievements->where('badge_level', 'gold')->count(),
                            'platinum' => $achievements->where('badge_level', 'platinum')->count(),
                        ],
                    ];
                })
                ->filter(function ($course) {
                    return $course['total_achievements'] > 0; // Only include courses with achievements
                })
                ->values();

            return response()->json([
                'courses' => $courses
            ]);
        } catch (\Exception $e) {
            \Log::error('Failed to fetch achievements by course: ' . $e->getMessage());
            return response()->json([
                'message' => 'Failed to fetch achievements by course',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Get achievements for a specific course
     */
    public function courseAchievements(Request $request, $courseId)
    {
        try {
            $userId = $request->query('user_id', 1);
            
            $course = Course::findOrFail($courseId);
            
            // Get all activities for this course
            $activityIds = Activity::whereHas('lesson', function($query) use ($courseId) {
                $query->where('course_id', $courseId);
            })->pluck('id');
            
            // Get achievements for these activities
            $achievements = ActivityCertificate::where('user_id', $userId)
                ->whereIn('activity_id', $activityIds)
                ->with(['activity', 'activity.lesson'])
                ->orderBy('earned_at', 'desc')
                ->get()
                ->map(function ($cert) {
                    $achievementData = is_string($cert->achievement_data) 
                        ? json_decode($cert->achievement_data, true) 
                        : $cert->achievement_data;
                    
                    return [
                        'id' => $cert->id,
                        'user_id' => $cert->user_id,
                        'activity_id' => $cert->activity_id,
                        'submission_id' => $cert->submission_id,
                        'certificate_id' => $cert->certificate_id,
                        'certificate_type' => $cert->certificate_type,
                        'badge_level' => $cert->badge_level,
                        'achievement_data' => array_merge($achievementData ?? [], [
                            'activity_title' => $cert->activity->title ?? 'Unknown Activity',
                            'lesson_title' => $cert->activity->lesson->title ?? 'Unknown Lesson',
                        ]),
                        'certificate_url' => $cert->certificate_url,
                        'is_verified' => $cert->is_verified,
                        'earned_at' => $cert->earned_at,
                        'created_at' => $cert->created_at,
                        'updated_at' => $cert->updated_at,
                    ];
                });
            
            // Get user progress for this course
            $progress = UserProgress::where('user_id', $userId)
                ->where('course_id', $courseId)
                ->get();
            
            $totalLessons = Lesson::where('course_id', $courseId)->count();
            $completedLessons = $progress->where('is_completed', 1)->count();
            
            $avgProgress = $progress->avg('completion_percentage') ?? 0;
            
            return response()->json([
                'course_id' => $course->id,
                'course_title' => $course->title,
                'course_slug' => $course->slug,
                'total_achievements' => $achievements->count(),
                'certificates_earned' => $achievements->count(),
                'progress_percentage' => round($avgProgress, 2),
                'achievements' => $achievements->values(),
                'lessons_completed' => $completedLessons,
                'total_lessons' => $totalLessons,
                'badges' => [
                    'bronze' => $achievements->where('badge_level', 'bronze')->count(),
                    'silver' => $achievements->where('badge_level', 'silver')->count(),
                    'gold' => $achievements->where('badge_level', 'gold')->count(),
                    'platinum' => $achievements->where('badge_level', 'platinum')->count(),
                ],
            ]);
        } catch (\Exception $e) {
            \Log::error('Failed to fetch course achievements: ' . $e->getMessage());
            return response()->json([
                'message' => 'Failed to fetch course achievements',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Get user progress
     */
    public function userProgress(Request $request)
    {
        try {
            $userId = $request->query('user_id', 1);
            
            $progress = UserProgress::where('user_id', $userId)
                ->with(['course', 'lesson'])
                ->orderBy('updated_at', 'desc')
                ->get()
                ->map(function ($p) {
                    $completedTopics = is_string($p->completed_topics) 
                        ? json_decode($p->completed_topics, true) 
                        : $p->completed_topics;
                    
                    $completedExercises = is_string($p->completed_exercises) 
                        ? json_decode($p->completed_exercises, true) 
                        : $p->completed_exercises;
                    
                    return [
                        'id' => $p->id,
                        'user_id' => $p->user_id,
                        'course_id' => $p->course_id,
                        'lesson_id' => $p->lesson_id,
                        'completion_percentage' => $p->completion_percentage,
                        'is_completed' => $p->is_completed,
                        'started_at' => $p->started_at,
                        'completed_at' => $p->completed_at,
                        'completed_topics' => $completedTopics ?? [],
                        'completed_exercises' => $completedExercises ?? [],
                        'score' => $p->score,
                        'created_at' => $p->created_at,
                        'updated_at' => $p->updated_at,
                    ];
                });

            return response()->json([
                'progress' => $progress
            ]);
        } catch (\Exception $e) {
            \Log::error('Failed to fetch user progress: ' . $e->getMessage());
            return response()->json([
                'message' => 'Failed to fetch user progress',
                'error' => $e->getMessage()
            ], 500);
        }
    }
}

