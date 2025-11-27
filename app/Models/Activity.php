<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;

class Activity extends Model
{
    use HasFactory;

    protected $fillable = [
        'lesson_id',
        'title',
        'description',
        'activity_type',
        'instructions',
        'questions',
        'resources',
        'time_limit',
        'max_attempts',
        'passing_score',
        'points',
        'order_index',
        'is_required',
        'is_published',
        'metadata'
    ];

    protected $casts = [
        'questions' => 'array',
        'resources' => 'array',
        'metadata' => 'array',
        'time_limit' => 'integer',
        'max_attempts' => 'integer',
        'passing_score' => 'integer',
        'points' => 'integer',
        'order_index' => 'integer',
    ];

    protected $attributes = [
        'is_required' => false,
        'is_published' => true,
        'points' => 0,
    ];

    /**
     * Set the is_required attribute
     */
    public function setIsRequiredAttribute($value)
    {
        $this->attributes['is_required'] = $value ? 1 : 0;
    }

    /**
     * Set the is_published attribute
     */
    public function setIsPublishedAttribute($value)
    {
        $this->attributes['is_published'] = $value ? 1 : 0;
    }

    /**
     * Get the is_required attribute
     */
    public function getIsRequiredAttribute($value)
    {
        return (bool) $value;
    }

    /**
     * Get the is_published attribute
     */
    public function getIsPublishedAttribute($value)
    {
        return (bool) $value;
    }

    protected static function boot()
    {
        parent::boot();

        static::creating(function ($activity) {
            // Auto-increment order_index
            if (is_null($activity->order_index)) {
                $maxOrder = static::where('lesson_id', $activity->lesson_id)
                    ->max('order_index');
                $activity->order_index = $maxOrder ? $maxOrder + 1 : 0;
            }
        });
    }

    /**
     * Get the lesson that owns the activity.
     */
    public function lesson(): BelongsTo
    {
        return $this->belongsTo(Lesson::class);
    }

    /**
     * Get all submissions for this activity.
     */
    public function submissions(): HasMany
    {
        return $this->hasMany(ActivitySubmission::class);
    }

    /**
     * Get only completed submissions for this activity.
     */
    public function completedSubmissions(): HasMany
    {
        return $this->hasMany(ActivitySubmission::class)->completed();
    }

    /**
     * Get only passed submissions for this activity.
     */
    public function passedSubmissions(): HasMany
    {
        return $this->hasMany(ActivitySubmission::class)->passed();
    }

    /**
     * Check if the activity is a coding activity
     */
    public function isCoding(): bool
    {
        return $this->activity_type === 'coding';
    }
}
