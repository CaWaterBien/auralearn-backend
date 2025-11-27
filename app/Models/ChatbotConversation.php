<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class ChatbotConversation extends Model
{
    use HasFactory;

    protected $fillable = [
        'session_id',
        'user_id',
        'message_id',
        'role',
        'content',
        'metadata',
        'html_context',
        'instructions_context',
        'retrieved_chunks',
        'tokens_used',
        'sent_at'
    ];

    protected $casts = [
        'metadata' => 'array',
        'retrieved_chunks' => 'array',
        'tokens_used' => 'integer',
        'sent_at' => 'datetime'
    ];

    /**
     * Get the user that owns the conversation
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    /**
     * Get conversation history for a session
     */
    public static function getSessionHistory(string $sessionId, int $limit = 50)
    {
        return self::where('session_id', $sessionId)
            ->orderBy('sent_at', 'asc')
            ->limit($limit)
            ->get();
    }

    /**
     * Get recent conversation context for RAG
     */
    public static function getRecentContext(string $sessionId, int $messageCount = 5)
    {
        return self::where('session_id', $sessionId)
            ->orderBy('sent_at', 'desc')
            ->limit($messageCount)
            ->get()
            ->reverse()
            ->values();
    }

    /**
     * Save user message with context
     */
    public static function saveUserMessage(
        string $sessionId, 
        string $messageId,
        string $content,
        ?int $userId = null,
        ?string $htmlContext = null,
        ?string $instructionsContext = null,
        array $metadata = []
    ) {
        return self::create([
            'session_id' => $sessionId,
            'user_id' => $userId,
            'message_id' => $messageId,
            'role' => 'user',
            'content' => $content,
            'metadata' => $metadata,
            'html_context' => $htmlContext,
            'instructions_context' => $instructionsContext,
            'sent_at' => now()
        ]);
    }

    /**
     * Save assistant message with RAG data
     */
    public static function saveAssistantMessage(
        string $sessionId,
        string $messageId,
        string $content,
        ?int $userId = null,
        array $retrievedChunks = [],
        ?int $tokensUsed = null,
        array $metadata = []
    ) {
        return self::create([
            'session_id' => $sessionId,
            'user_id' => $userId,
            'message_id' => $messageId,
            'role' => 'assistant',
            'content' => $content,
            'metadata' => $metadata,
            'retrieved_chunks' => $retrievedChunks,
            'tokens_used' => $tokensUsed,
            'sent_at' => now()
        ]);
    }

    /**
     * Get conversation analytics
     */
    public static function getAnalytics(string $sessionId)
    {
        $stats = self::where('session_id', $sessionId)
            ->selectRaw('
                COUNT(*) as total_messages,
                COUNT(CASE WHEN role = "user" THEN 1 END) as user_messages,
                COUNT(CASE WHEN role = "assistant" THEN 1 END) as assistant_messages,
                SUM(tokens_used) as total_tokens,
                MIN(sent_at) as first_message,
                MAX(sent_at) as last_message
            ')
            ->first();

        return $stats;
    }
}

