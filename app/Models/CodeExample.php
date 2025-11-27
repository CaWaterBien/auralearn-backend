<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class CodeExample extends Model
{
    use HasFactory;

    protected $fillable = [
        'lesson_id',
        'topic_id',
        'title',
        'description',
        'language',
        'initial_code',
        'solution_code',
        'hints',
        'is_interactive',
        'show_preview',
        'allow_reset',
        'test_cases',
        'order_index'
    ];

    protected $casts = [
        'test_cases' => 'array',
        'order_index' => 'integer',
    ];

    protected $attributes = [
        'is_interactive' => true,
        'show_preview' => true,
        'allow_reset' => true,
    ];

    protected static function boot()
    {
        parent::boot();

        static::creating(function ($example) {
            // Auto-increment order_index based on parent (lesson or topic)
            if (is_null($example->order_index)) {
                $query = static::query();
                
                if ($example->topic_id) {
                    $query->where('topic_id', $example->topic_id);
                } elseif ($example->lesson_id) {
                    $query->where('lesson_id', $example->lesson_id)
                          ->whereNull('topic_id');
                }
                
                $maxOrder = $query->max('order_index');
                $example->order_index = $maxOrder ? $maxOrder + 1 : 0;
            }
        });
    }

    /**
     * Get the lesson that owns the code example.
     */
    public function lesson(): BelongsTo
    {
        return $this->belongsTo(Lesson::class);
    }

    /**
     * Get the topic that owns the code example.
     */
    public function topic(): BelongsTo
    {
        return $this->belongsTo(Topic::class);
    }

    /**
     * Get the parent (lesson or topic) title
     */
    public function getParentTitle(): string
    {
        if ($this->topic) {
            return $this->topic->title;
        }
        if ($this->lesson) {
            return $this->lesson->title;
        }
        return '';
    }
}
