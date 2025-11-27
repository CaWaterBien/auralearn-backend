<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class UserProgress extends Model
{
    use HasFactory;

    protected $table = 'user_progress';

    protected $fillable = [
        'user_id',
        'course_id',
        'lesson_id',
        'completion_percentage',
        'is_completed',
        'started_at',
        'completed_at',
        'completed_topics',
        'completed_exercises',
        'score'
    ];

    protected $casts = [
        'completion_percentage' => 'decimal:2',
        'completed_topics' => 'array',
        'completed_exercises' => 'array',
        'score' => 'integer',
        'is_completed' => 'boolean',
        'started_at' => 'datetime',
        'completed_at' => 'datetime',
    ];

    protected $attributes = [
        'is_completed' => false,
        'completion_percentage' => 0,
    ];

    /**
     * Get the user that owns the progress.
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    /**
     * Get the course for this progress.
     */
    public function course(): BelongsTo
    {
        return $this->belongsTo(Course::class);
    }

    /**
     * Get the lesson for this progress.
     */
    public function lesson(): BelongsTo
    {
        return $this->belongsTo(Lesson::class);
    }

    /**
     * Mark a topic as completed
     */
    public function markTopicCompleted($topicId): void
    {
        $completedTopics = $this->completed_topics ?? [];
        
        if (!in_array($topicId, $completedTopics)) {
            $completedTopics[] = $topicId;
            $this->completed_topics = $completedTopics;
            $this->updateCompletionPercentage();
            $this->save();
        }
    }

    /**
     * Mark an exercise as completed
     */
    public function markExerciseCompleted($exerciseId): void
    {
        $completedExercises = $this->completed_exercises ?? [];
        
        if (!in_array($exerciseId, $completedExercises)) {
            $completedExercises[] = $exerciseId;
            $this->completed_exercises = $completedExercises;
            $this->updateCompletionPercentage();
            $this->save();
        }
    }

    /**
     * Update the completion percentage based on completed items
     */
    public function updateCompletionPercentage(): void
    {
        if ($this->lesson) {
            $totalTopics = $this->lesson->topics()->count();
            $totalExercises = $this->lesson->codeExamples()->count();
            $totalItems = $totalTopics + $totalExercises;
            
            if ($totalItems > 0) {
                $completedItems = count($this->completed_topics ?? []) + 
                                  count($this->completed_exercises ?? []);
                $percentage = ($completedItems / $totalItems) * 100;
                
                $this->completion_percentage = min(100, $percentage);
                
                if ($this->completion_percentage >= 100) {
                    $this->is_completed = true;
                    $this->completed_at = now();
                }
            }
        }
    }

    /**
     * Start the progress if not already started
     */
    public function startIfNotStarted(): void
    {
        if (!$this->started_at) {
            $this->started_at = now();
            $this->save();
        }
    }
}
