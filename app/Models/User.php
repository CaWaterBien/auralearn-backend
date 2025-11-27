<?php

namespace App\Models;

// use Illuminate\Contracts\Auth\MustVerifyEmail;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Foundation\Auth\User as Authenticatable;
use Illuminate\Notifications\Notifiable;
use Illuminate\Database\Eloquent\Relations\HasMany;

class User extends Authenticatable
{
    /** @use HasFactory<\Database\Factories\UserFactory> */
    use HasFactory, Notifiable;

    /**
     * The attributes that are mass assignable.
     *
     * @var list<string>
     */
    protected $fillable = [
        'name',
        'full_name',
        'email',
        'password',
        'avatar',
        'join_date',
        'progress',
        'preferences',
        'is_active',
    ];

    /**
     * The attributes that should be hidden for serialization.
     *
     * @var list<string>
     */
    protected $hidden = [
        'password',
        'remember_token',
    ];

    /**
     * Get the attributes that should be cast.
     *
     * @return array<string, string>
     */
    protected function casts(): array
    {
        return [
            'email_verified_at' => 'datetime',
            'password' => 'hashed',
            'join_date' => 'date',
            'progress' => 'array',
            'preferences' => 'array',
            'is_active' => 'integer',
        ];
    }

    /**
     * Get the user's default progress data.
     */
    public function getDefaultProgress(): array
    {
        return [
            'completedCourses' => 0,
            'totalCourses' => 15,
            'currentCourse' => 'HTML Fundamentals',
            'streak' => 0,
            'totalPoints' => 0,
            'rank' => 999,
            'badges' => ['Welcome Badge'],
            'recentLessons' => []
        ];
    }

    /**
     * Get the user's default preferences.
     */
    public function getDefaultPreferences(): array
    {
        return [
            // General Settings
            'theme' => 'light',
            'language' => 'en',
            
            // Notifications Settings
            'emailNotifications' => true,
            'pushNotifications' => true,
            'weeklyDigest' => true,
            'achievementAlerts' => true,
            'reminderAlerts' => true,
            
            // Privacy Settings
            'profileVisibility' => 'public',
            'showProgress' => true,
            'showBadges' => true,
            'allowMessages' => true,
            
            // Learning Settings
            'dailyGoal' => 30,
            'difficultyLevel' => 'intermediate',
            'autoSave' => true,
            'skipIntros' => false,
            'showHints' => true,
            
            // Accessibility Settings
            'fontSize' => 'medium',
            'reducedMotion' => false,
            'highContrast' => false,
            'soundEffects' => true,
        ];
    }
    
    /**
     * Calculate user's storage usage in bytes.
     */
    public function calculateStorageUsage(): int
    {
        $size = 0;
        
        // Calculate size of user data
        $size += strlen(json_encode($this->progress ?? []));
        $size += strlen(json_encode($this->preferences ?? []));
        $size += strlen($this->avatar ?? '');
        
        // Calculate size of activity submissions
        $submissions = $this->activitySubmissions()->get();
        foreach ($submissions as $submission) {
            $size += strlen(json_encode($submission->toArray()));
        }
        
        return $size;
    }

    /**
     * Get all activity submissions for this user.
     */
    public function activitySubmissions(): HasMany
    {
        return $this->hasMany(ActivitySubmission::class);
    }

    /**
     * Get only completed activity submissions for this user.
     */
    public function completedActivitySubmissions(): HasMany
    {
        return $this->hasMany(ActivitySubmission::class)->completed();
    }

    /**
     * Get user's submission for a specific activity.
     */
    public function submissionForActivity($activityId)
    {
        return $this->activitySubmissions()
            ->where('activity_id', $activityId)
            ->orderBy('attempt_number', 'desc')
            ->first();
    }

    /**
     * Get all attempts for a specific activity.
     */
    public function allAttemptsForActivity($activityId)
    {
        return $this->activitySubmissions()
            ->where('activity_id', $activityId)
            ->orderBy('attempt_number', 'asc')
            ->get();
    }

    /**
     * Check if user has completed a specific activity.
     */
    public function hasCompletedActivity($activityId): bool
    {
        return $this->activitySubmissions()
            ->where('activity_id', $activityId)
            ->where('is_completed', true)
            ->exists();
    }
}
