<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;

class Topic extends Model
{
    use HasFactory;

    protected $fillable = [
        'lesson_id',
        'title',
        'content',
        'content_type',
        'order_index',
        'metadata'
    ];

    protected $casts = [
        'metadata' => 'array',
        'order_index' => 'integer',
    ];

    protected static function boot()
    {
        parent::boot();

        static::creating(function ($topic) {
            // Auto-increment order_index
            if (is_null($topic->order_index)) {
                $maxOrder = static::where('lesson_id', $topic->lesson_id)
                    ->max('order_index');
                $topic->order_index = $maxOrder ? $maxOrder + 1 : 0;
            }
        });
    }

    /**
     * Get the lesson that owns the topic.
     */
    public function lesson(): BelongsTo
    {
        return $this->belongsTo(Lesson::class);
    }

    /**
     * Get the code examples for the topic.
     */
    public function codeExamples(): HasMany
    {
        return $this->hasMany(CodeExample::class)->orderBy('order_index');
    }
}
