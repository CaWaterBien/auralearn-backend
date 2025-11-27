<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Support\Facades\DB;

class ChatbotSession extends Model
{
    use HasFactory;

    protected $fillable = [
        'session_id',
        'user_id',
        'attempt_count',
        'max_attempts',
        'is_blocked',
        'last_activity',
        'blocked_until',
        'progress_data'
    ];

    protected $casts = [
        'attempt_count' => 'integer',
        'max_attempts' => 'integer',
        'is_blocked' => 'boolean',
        'last_activity' => 'datetime',
        'blocked_until' => 'datetime',
        'progress_data' => 'array'
    ];

    protected $attributes = [
        'is_blocked' => false,
        'attempt_count' => 0,
        'progress_data' => '[]'
    ];

    /**
     * Get the user that owns the session
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    /**
     * Get conversations for this session
     */
    public function conversations(): HasMany
    {
        return $this->hasMany(ChatbotConversation::class, 'session_id', 'session_id');
    }

    /**
     * Get or create session
     */
    public static function getOrCreate(string $sessionId, ?int $userId = null): self
    {
        // Use raw DB insert to handle boolean properly
        $existing = self::where('session_id', $sessionId)->first();
        if ($existing) {
            return $existing;
        }

        // Use raw SQL insert with explicit PostgreSQL boolean casting
        DB::statement("
            INSERT INTO chatbot_sessions 
            (session_id, user_id, attempt_count, max_attempts, is_blocked, last_activity, progress_data, created_at, updated_at) 
            VALUES (?, ?, ?, ?, ?::boolean, ?, ?::json, ?, ?)
        ", [
            $sessionId,
            $userId,
            0,
            (int) env('AURABOT_ATTEMPT_LIMIT', 3),
            'false', // PostgreSQL boolean string
            now(),
            '[]',
            now(),
            now()
        ]);

        // Return the created session
        return self::where('session_id', $sessionId)->first();
    }

    /**
     * Check if user can ask a question
     */
    public function canAskQuestion(): bool
    {
        // Check if blocked and if block period has expired
        if ($this->is_blocked && $this->blocked_until && $this->blocked_until->isFuture()) {
            return false;
        }

        // If block period expired, reset the session
        if ($this->is_blocked && $this->blocked_until && $this->blocked_until->isPast()) {
            $this->resetAttempts();
            return true;
        }

        // Check attempt limit
        return $this->attempt_count < $this->max_attempts;
    }

    /**
     * Increment attempt count
     */
    public function incrementAttempt(): void
    {
        $this->increment('attempt_count');
        $this->update(['last_activity' => now()]);

        // Block if reached limit
        if ($this->attempt_count >= $this->max_attempts) {
            $this->blockSession();
        }
    }

    /**
     * Block session for a period
     */
    public function blockSession(int $hours = 1): void
    {
        $this->update([
            'is_blocked' => true,
            'blocked_until' => now()->addHours($hours)
        ]);
    }

    /**
     * Reset attempts and unblock
     */
    public function resetAttempts(): void
    {
        $this->update([
            'attempt_count' => 0,
            'is_blocked' => false,
            'blocked_until' => null,
            'last_activity' => now()
        ]);
    }

    /**
     * Update progress data
     */
    public function updateProgress(array $progressData): void
    {
        $currentProgress = $this->progress_data ?? [];
        $mergedProgress = array_merge($currentProgress, $progressData);
        
        $this->update([
            'progress_data' => $mergedProgress,
            'last_activity' => now()
        ]);
    }

    /**
     * Get remaining attempts
     */
    public function getRemainingAttempts(): int
    {
        if ($this->is_blocked) {
            return 0;
        }
        
        return max(0, $this->max_attempts - $this->attempt_count);
    }

    /**
     * Get time until unblocked
     */
    public function getTimeUntilUnblocked(): ?string
    {
        if (!$this->is_blocked || !$this->blocked_until) {
            return null;
        }

        if ($this->blocked_until->isPast()) {
            return null;
        }

        return $this->blocked_until->diffForHumans();
    }
}

