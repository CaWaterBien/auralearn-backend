<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Support\Str;

class Lesson extends Model
{
    use HasFactory;

    protected $fillable = [
        'course_id',
        'title',
        'slug',
        'description',
        'content',
        'order_index',
        'duration_minutes',
        'is_locked',
        'is_published',
        'lesson_type',
        'objectives',
        'prerequisites'
    ];

    protected $casts = [
        'objectives' => 'array',
        'prerequisites' => 'array',
        'order_index' => 'integer',
        'duration_minutes' => 'integer',
    ];

    protected $attributes = [
        'is_locked' => false,
        'is_published' => true,
    ];

    protected static function boot()
    {
        parent::boot();

        static::creating(function ($lesson) {
            if (empty($lesson->slug)) {
                $lesson->slug = Str::slug($lesson->title);
            }
            
            // Auto-increment order_index
            if (is_null($lesson->order_index)) {
                $maxOrder = static::where('course_id', $lesson->course_id)
                    ->max('order_index');
                $lesson->order_index = $maxOrder ? $maxOrder + 1 : 0;
            }
        });

        static::updating(function ($lesson) {
            if ($lesson->isDirty('title') && !$lesson->isDirty('slug')) {
                $lesson->slug = Str::slug($lesson->title);
            }
        });

        // Update course lesson count when lesson is created or deleted
        static::created(function ($lesson) {
            $lesson->course->updateLessonCount();
        });

        static::deleted(function ($lesson) {
            $lesson->course->updateLessonCount();
        });
    }

    /**
     * Get the course that owns the lesson.
     */
    public function course(): BelongsTo
    {
        return $this->belongsTo(Course::class);
    }

    /**
     * Get the topics for the lesson.
     */
    public function topics(): HasMany
    {
        return $this->hasMany(Topic::class)->orderBy('order_index');
    }

    /**
     * Get the code examples for the lesson.
     */
    public function codeExamples(): HasMany
    {
        return $this->hasMany(CodeExample::class)->orderBy('order_index');
    }

    /**
     * Get the activities for the lesson.
     */
    public function activities(): HasMany
    {
        return $this->hasMany(Activity::class)->orderBy('order_index');
    }

    /**
     * Get the user progress for this lesson.
     */
    public function userProgress(): HasMany
    {
        return $this->hasMany(UserProgress::class);
    }

    /**
     * Check if user has completed this lesson
     */
    public function isCompletedByUser($userId): bool
    {
        return $this->userProgress()
            ->where('user_id', $userId)
            ->where('is_completed', true)
            ->exists();
    }

    /**
     * Get next lesson in the course
     */
    public function getNextLesson()
    {
        return self::where('course_id', $this->course_id)
            ->where('order_index', '>', $this->order_index)
            ->orderBy('order_index')
            ->first();
    }

    /**
     * Get previous lesson in the course
     */
    public function getPreviousLesson()
    {
        return self::where('course_id', $this->course_id)
            ->where('order_index', '<', $this->order_index)
            ->orderBy('order_index', 'desc')
            ->first();
    }
}
