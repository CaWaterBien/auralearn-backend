<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Support\Str;

class ActivityCertificate extends Model
{
    use HasFactory;

    protected $fillable = [
        'user_id',
        'activity_id',
        'submission_id',
        'certificate_id',
        'certificate_type',
        'badge_level',
        'achievement_data',
        'certificate_url',
        'is_verified',
        'earned_at'
    ];

    protected $casts = [
        'achievement_data' => 'array',
        'is_verified' => 'boolean',
        'earned_at' => 'datetime'
    ];

    /**
     * Get the user who earned this certificate
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    /**
     * Get the activity this certificate is for
     */
    public function activity(): BelongsTo
    {
        return $this->belongsTo(Activity::class);
    }

    /**
     * Get the submission that earned this certificate
     */
    public function submission(): BelongsTo
    {
        return $this->belongsTo(ActivitySubmission::class, 'submission_id');
    }

    /**
     * Generate a unique certificate ID
     */
    public static function generateCertificateId(): string
    {
        do {
            $id = 'CERT-' . strtoupper(Str::random(8));
        } while (self::where('certificate_id', $id)->exists());
        
        return $id;
    }

    /**
     * Award a certificate to a user for completing an activity
     */
    public static function awardCertificate(
        $userId,
        $activityId,
        $submissionId,
        $certificateType,
        $achievementData
    ): ?self {
        // Check if user already has this type of certificate for this activity
        $existing = self::where([
            'user_id' => $userId,
            'activity_id' => $activityId,
            'certificate_type' => $certificateType
        ])->first();

        if ($existing) {
            return $existing; // Don't create duplicates
        }

        // Determine badge level based on achievement data
        $badgeLevel = self::determineBadgeLevel($achievementData, $certificateType);

        // Create certificate
        return self::create([
            'user_id' => $userId,
            'activity_id' => $activityId,
            'submission_id' => $submissionId,
            'certificate_id' => self::generateCertificateId(),
            'certificate_type' => $certificateType,
            'badge_level' => $badgeLevel,
            'achievement_data' => $achievementData,
            'is_verified' => true,
            'earned_at' => now()
        ]);
    }

    /**
     * Determine badge level based on achievement data
     */
    private static function determineBadgeLevel($achievementData, $certificateType): string
    {
        $score = $achievementData['score'] ?? 0;
        $attemptNumber = $achievementData['attempt_number'] ?? 999;
        $timeSpent = $achievementData['time_spent_minutes'] ?? 999;

        // Perfect score criteria
        if ($score === 100 && $certificateType === 'perfect_score') {
            return 'platinum';
        }

        // First attempt success criteria
        if ($attemptNumber === 1 && $score >= 90) {
            return 'gold';
        }

        // Excellence criteria (high score with reasonable attempts/time)
        if ($score >= 95 && $attemptNumber <= 2) {
            return 'gold';
        } elseif ($score >= 90 && $attemptNumber <= 3) {
            return 'silver';
        }

        // Quick completion bonus
        if ($score >= 80 && $timeSpent <= 15) {
            return $score >= 90 ? 'silver' : 'bronze';
        }

        // Standard completion
        if ($score >= 80) {
            return 'bronze';
        }

        return 'bronze'; // Default
    }

    /**
     * Check if user qualifies for certificates based on submission
     */
    public static function checkAndAwardCertificates($submission): array
    {
        if (!$submission->is_completed) {
            return [];
        }

        $certificates = [];
        $achievementData = [
            'score' => $submission->score,
            'attempt_number' => $submission->attempt_number,
            'time_spent_minutes' => $submission->time_spent_minutes,
            'validation_results' => $submission->validation_results,
            'similarity_check' => $submission->validation_results['similarity_check'] ?? null,
            'explanation_analysis' => $submission->explanation_analysis ?? null
        ];

        // Completion Certificate (everyone who completes gets this)
        if ($submission->score >= 80) {
            $cert = self::awardCertificate(
                $submission->user_id,
                $submission->activity_id,
                $submission->id,
                'completion',
                $achievementData
            );
            if ($cert) $certificates[] = $cert;
        }

        // Excellence Certificate (score >= 90)
        if ($submission->score >= 90) {
            $cert = self::awardCertificate(
                $submission->user_id,
                $submission->activity_id,
                $submission->id,
                'excellence',
                $achievementData
            );
            if ($cert) $certificates[] = $cert;
        }

        // First Attempt Certificate (completed on first try with good score)
        if ($submission->attempt_number === 1 && $submission->score >= 85) {
            $cert = self::awardCertificate(
                $submission->user_id,
                $submission->activity_id,
                $submission->id,
                'first_attempt',
                $achievementData
            );
            if ($cert) $certificates[] = $cert;
        }

        // Perfect Score Certificate (100% score)
        if ($submission->score === 100) {
            $cert = self::awardCertificate(
                $submission->user_id,
                $submission->activity_id,
                $submission->id,
                'perfect_score',
                $achievementData
            );
            if ($cert) $certificates[] = $cert;
        }

        return $certificates;
    }

    /**
     * Get certificate display information
     */
    public function getDisplayInfo(): array
    {
        $types = [
            'completion' => [
                'title' => 'Completion Certificate',
                'description' => 'Successfully completed the activity',
                'icon' => '✅'
            ],
            'excellence' => [
                'title' => 'Excellence Award',
                'description' => 'Achieved excellent performance',
                'icon' => '🏆'
            ],
            'first_attempt' => [
                'title' => 'First Attempt Success',
                'description' => 'Completed on the first try',
                'icon' => '🎯'
            ],
            'perfect_score' => [
                'title' => 'Perfect Score',
                'description' => 'Achieved 100% score',
                'icon' => '💎'
            ]
        ];

        $levels = [
            'bronze' => ['color' => '#CD7F32', 'label' => 'Bronze'],
            'silver' => ['color' => '#C0C0C0', 'label' => 'Silver'],
            'gold' => ['color' => '#FFD700', 'label' => 'Gold'],
            'platinum' => ['color' => '#E5E4E2', 'label' => 'Platinum']
        ];

        return [
            'certificate_id' => $this->certificate_id,
            'type' => $types[$this->certificate_type] ?? $types['completion'],
            'badge_level' => $levels[$this->badge_level] ?? $levels['bronze'],
            'earned_at' => $this->earned_at->format('F j, Y'),
            'achievement_data' => $this->achievement_data
        ];
    }
}
