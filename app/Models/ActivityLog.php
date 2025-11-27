<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Support\Facades\Request;

class ActivityLog extends Model
{
    use HasFactory;

    protected $fillable = [
        'user_id',
        'activity_id',
        'submission_id',
        'event_type',
        'event_data',
        'ip_address',
        'user_agent',
        'created_at'
    ];

    protected $casts = [
        'event_data' => 'array',
        'created_at' => 'datetime'
    ];

    public $timestamps = false; // Only using created_at

    /**
     * Get the user that performed this action
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    /**
     * Get the activity this log is for
     */
    public function activity(): BelongsTo
    {
        return $this->belongsTo(Activity::class);
    }

    /**
     * Get the submission this log is related to
     */
    public function submission(): BelongsTo
    {
        return $this->belongsTo(ActivitySubmission::class, 'submission_id');
    }

    /**
     * Create a new activity log entry
     */
    public static function logEvent(
        $userId,
        $activityId,
        $eventType,
        $eventData = [],
        $submissionId = null
    ): self {
        return self::create([
            'user_id' => $userId,
            'activity_id' => $activityId,
            'submission_id' => $submissionId,
            'event_type' => $eventType,
            'event_data' => $eventData,
            'ip_address' => Request::ip(),
            'user_agent' => Request::userAgent(),
            'created_at' => now()
        ]);
    }

    /**
     * Log activity started
     */
    public static function logActivityStarted($userId, $activityId, $additionalData = []): self
    {
        return self::logEvent($userId, $activityId, 'activity_started', array_merge([
            'timestamp' => now()->toISOString(),
            'initial_load' => true
        ], $additionalData));
    }

    /**
     * Log code change
     */
    public static function logCodeChanged($userId, $activityId, $codeLength, $additionalData = []): self
    {
        return self::logEvent($userId, $activityId, 'code_changed', array_merge([
            'code_length' => $codeLength,
            'timestamp' => now()->toISOString()
        ], $additionalData));
    }

    /**
     * Log validation attempt
     */
    public static function logValidationAttempted($userId, $activityId, $validationResults, $submissionId = null): self
    {
        return self::logEvent($userId, $activityId, 'validation_attempted', [
            'validation_results' => $validationResults,
            'timestamp' => now()->toISOString(),
            'success' => $validationResults['overall']['percentage'] ?? 0 >= 80
        ], $submissionId);
    }

    /**
     * Log submission created
     */
    public static function logSubmissionCreated($userId, $activityId, $submissionData, $submissionId): self
    {
        return self::logEvent($userId, $activityId, 'submission_created', [
            'attempt_number' => $submissionData['attempt_number'],
            'score' => $submissionData['score'],
            'completion_status' => $submissionData['completion_status'],
            'time_spent_minutes' => $submissionData['time_spent_minutes'],
            'code_length' => strlen($submissionData['submitted_code']),
            'timestamp' => now()->toISOString()
        ], $submissionId);
    }

    /**
     * Log hint requested
     */
    public static function logHintRequested($userId, $activityId, $hintNumber, $totalHints): self
    {
        return self::logEvent($userId, $activityId, 'hint_requested', [
            'hint_number' => $hintNumber,
            'total_hints' => $totalHints,
            'timestamp' => now()->toISOString()
        ]);
    }

    /**
     * Log activity completed
     */
    public static function logActivityCompleted($userId, $activityId, $submissionData, $submissionId): self
    {
        return self::logEvent($userId, $activityId, 'activity_completed', [
            'final_score' => $submissionData['score'],
            'attempt_number' => $submissionData['attempt_number'],
            'completion_status' => $submissionData['completion_status'],
            'time_spent_total' => $submissionData['time_spent_minutes'],
            'timestamp' => now()->toISOString()
        ], $submissionId);
    }

    /**
     * Log activity reset
     */
    public static function logActivityReset($userId, $activityId): self
    {
        return self::logEvent($userId, $activityId, 'activity_reset', [
            'timestamp' => now()->toISOString()
        ]);
    }

    /**
     * Log instructor override
     */
    public static function logInstructorOverride($adminId, $submissionId, $overrideData): self
    {
        $submission = ActivitySubmission::find($submissionId);
        
        return self::logEvent($submission->user_id, $submission->activity_id, 'instructor_override', [
            'admin_id' => $adminId,
            'old_status' => $overrideData['old_status'],
            'new_status' => $overrideData['new_status'],
            'old_score' => $overrideData['old_score'],
            'new_score' => $overrideData['new_score'],
            'instructor_notes' => $overrideData['instructor_notes'] ?? null,
            'timestamp' => now()->toISOString()
        ], $submissionId);
    }

    /**
     * Get activity statistics from logs
     */
    public static function getActivityStats($activityId, $days = 30)
    {
        $since = now()->subDays($days);
        
        return [
            'total_interactions' => self::where('activity_id', $activityId)
                ->where('created_at', '>=', $since)->count(),
            
            'unique_users' => self::where('activity_id', $activityId)
                ->where('created_at', '>=', $since)
                ->distinct('user_id')->count('user_id'),
                
            'completion_rate' => self::getCompletionRate($activityId, $since),
            
            'average_attempts' => self::getAverageAttempts($activityId, $since),
            
            'hint_usage' => self::where('activity_id', $activityId)
                ->where('event_type', 'hint_requested')
                ->where('created_at', '>=', $since)->count(),
                
            'reset_count' => self::where('activity_id', $activityId)
                ->where('event_type', 'activity_reset')
                ->where('created_at', '>=', $since)->count()
        ];
    }

    /**
     * Calculate completion rate
     */
    private static function getCompletionRate($activityId, $since)
    {
        $totalStarted = self::where('activity_id', $activityId)
            ->where('event_type', 'activity_started')
            ->where('created_at', '>=', $since)
            ->distinct('user_id')->count('user_id');
            
        $totalCompleted = self::where('activity_id', $activityId)
            ->where('event_type', 'activity_completed')
            ->where('created_at', '>=', $since)
            ->distinct('user_id')->count('user_id');
            
        return $totalStarted > 0 ? round(($totalCompleted / $totalStarted) * 100, 2) : 0;
    }

    /**
     * Calculate average attempts before completion
     */
    private static function getAverageAttempts($activityId, $since)
    {
        $completedSubmissions = ActivitySubmission::where('activity_id', $activityId)
            ->where('is_completed', true)
            ->where('created_at', '>=', $since)
            ->get();
            
        if ($completedSubmissions->isEmpty()) {
            return 0;
        }
        
        return round($completedSubmissions->avg('attempt_number'), 2);
    }
}
