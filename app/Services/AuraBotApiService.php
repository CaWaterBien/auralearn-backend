<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class AuraBotApiService
{
    private string $baseUrl;

    public function __construct()
    {
        $this->baseUrl = env('NEXT_PUBLIC_API_BASE', 'http://localhost:8000');
    }

    /**
     * Send question to AuraBot backend
     */
    public function askQuestion(
        string $sessionId,
        string $question,
        ?string $htmlContext = null,
        ?string $instructionsContext = null,
        ?int $userId = null
    ): array {
        try {
            $response = Http::timeout(60)
                ->post("{$this->baseUrl}/api/aurabot/ask", [
                    'session_id' => $sessionId,
                    'question' => $question,
                    'html_context' => $htmlContext,
                    'instructions_context' => $instructionsContext,
                    'user_id' => $userId
                ]);

            if ($response->successful()) {
                return $response->json();
            }

            Log::error('AuraBot API error', [
                'status' => $response->status(),
                'response' => $response->body()
            ]);

            return [
                'success' => false,
                'error' => 'Failed to get response from AuraBot service'
            ];

        } catch (\Exception $e) {
            Log::error('AuraBot API exception', [
                'error' => $e->getMessage()
            ]);

            return [
                'success' => false,
                'error' => 'Connection error with AuraBot service'
            ];
        }
    }

    /**
     * Get session status
     */
    public function getSessionStatus(string $sessionId): array
    {
        try {
            $response = Http::timeout(30)
                ->get("{$this->baseUrl}/api/aurabot/session-status", [
                    'session_id' => $sessionId
                ]);

            if ($response->successful()) {
                return $response->json();
            }

            return [
                'exists' => false,
                'can_ask' => true,
                'remaining_attempts' => 3,
                'attempt_count' => 0
            ];

        } catch (\Exception $e) {
            Log::error('Session status API exception', [
                'error' => $e->getMessage()
            ]);

            return [
                'exists' => false,
                'can_ask' => true,
                'remaining_attempts' => 3,
                'attempt_count' => 0
            ];
        }
    }

    /**
     * Get conversation history
     */
    public function getConversationHistory(string $sessionId, int $limit = 20): array
    {
        try {
            $response = Http::timeout(30)
                ->get("{$this->baseUrl}/api/aurabot/conversation-history", [
                    'session_id' => $sessionId,
                    'limit' => $limit
                ]);

            if ($response->successful()) {
                $data = $response->json();
                return $data['messages'] ?? [];
            }

            return [];

        } catch (\Exception $e) {
            Log::error('Conversation history API exception', [
                'error' => $e->getMessage()
            ]);

            return [];
        }
    }
}

