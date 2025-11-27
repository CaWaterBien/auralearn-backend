<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Support\Str;

class Course extends Model
{
    use HasFactory;

    protected $fillable = [
        'title',
        'slug',
        'description',
        'category',
        'difficulty_level',
        'total_lessons',
        'duration_hours',
        'tags',
        'thumbnail',
        'is_free',
        'is_published',
        'order_index',
        'metadata'
    ];

    protected $casts = [
        'tags' => 'array',
        'metadata' => 'array',
        'total_lessons' => 'integer',
        'duration_hours' => 'integer',
        'order_index' => 'integer',
        'is_published' => 'boolean',
        'is_free' => 'boolean',
    ];

    protected $attributes = [
        'is_free' => true,
        'is_published' => false,
    ];

    /**
     * Set the is_free attribute
     */
    public function setIsFreeAttribute($value)
    {
        $this->attributes['is_free'] = $value ? true : false;
    }

    /**
     * Set the is_published attribute
     */
    public function setIsPublishedAttribute($value)
    {
        $this->attributes['is_published'] = $value ? true : false;
    }

    /**
     * Get the is_free attribute
     */
    public function getIsFreeAttribute($value)
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

        static::creating(function ($course) {
            if (empty($course->slug)) {
                $course->slug = Str::slug($course->title);
            }
        });

        static::updating(function ($course) {
            if ($course->isDirty('title') && !$course->isDirty('slug')) {
                $course->slug = Str::slug($course->title);
            }
        });
    }

    /**
     * Get the lessons for the course.
     */
    public function lessons(): HasMany
    {
        return $this->hasMany(Lesson::class)->orderBy('order_index');
    }

    /**
     * Get published lessons for the course.
     */
    public function publishedLessons(): HasMany
    {
        return $this->hasMany(Lesson::class)
            ->where('is_published', true)
            ->orderBy('order_index');
    }

    /**
     * Get the user progress for this course.
     */
    public function userProgress(): HasMany
    {
        return $this->hasMany(UserProgress::class);
    }

    /**
     * Calculate total duration from lessons
     */
    public function calculateTotalDuration(): int
    {
        return $this->lessons()->sum('duration_minutes');
    }

    /**
     * Update lesson count
     */
    public function updateLessonCount(): void
    {
        $this->update([
            'total_lessons' => $this->lessons()->count(),
            'duration_hours' => round($this->calculateTotalDuration() / 60, 1)
        ]);
    }
}
