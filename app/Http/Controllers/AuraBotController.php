<?php

namespace App\Http\Controllers;

use App\Services\AuraBotRagService;
use App\Services\NebiusClient;
use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Validator;

class AuraBotController extends Controller
{
    private AuraBotRagService $auraBotService;

    public function __construct(AuraBotRagService $auraBotService)
    {
        $this->auraBotService = $auraBotService;
    }

    /**
     * Handle chatbot question from frontend
     */
    public function askQuestion(Request $request): JsonResponse
    {
        $validator = Validator::make($request->all(), [
            'session_id' => 'required|string|max:255',
            'question' => 'required|string|max:2000',
            'html_context' => 'nullable|string|max:10000',
            'instructions_context' => 'nullable|string|max:5000',
            'feedback_context' => 'nullable|string|max:5000',
            'user_id' => 'nullable|integer'  // Removed exists:users,id for now
        ]);

        if ($validator->fails()) {
            Log::warning('AuraBot validation failed', [
                'errors' => $validator->errors(),
                'request_data' => $request->all()
            ]);
            
            return response()->json([
                'success' => false,
                'error' => 'Invalid input data',
                'details' => $validator->errors()
            ], 422);
        }

        try {
            $result = $this->auraBotService->processUserQuestion(
                $request->input('session_id'),
                $request->input('question'),
                $request->input('html_context'),
                $request->input('instructions_context'),
                $request->input('feedback_context'),
                $request->input('user_id')
            );

            Log::info('AuraBot question processed', [
                'session_id' => $request->input('session_id'),
                'success' => $result['success'],
                'tokens_used' => $result['tokens_used'] ?? 0
            ]);

            return response()->json($result);

        } catch (\Exception $e) {
            Log::error('AuraBot controller error', [
                'error' => $e->getMessage(),
                'session_id' => $request->input('session_id')
            ]);

            return response()->json([
                'success' => false,
                'error' => 'An unexpected error occurred. Please try again.'
            ], 500);
        }
    }

    /**
     * Get session status and remaining attempts
     */
    public function getSessionStatus(Request $request): JsonResponse
    {
        $validator = Validator::make($request->all(), [
            'session_id' => 'required|string|max:255'
        ]);

        if ($validator->fails()) {
            return response()->json([
                'success' => false,
                'error' => 'Invalid session ID'
            ], 422);
        }

        try {
            $status = $this->auraBotService->getSessionStatus($request->input('session_id'));
            return response()->json($status);

        } catch (\Exception $e) {
            Log::error('Session status error', [
                'error' => $e->getMessage(),
                'session_id' => $request->input('session_id')
            ]);

            return response()->json([
                'success' => false,
                'error' => 'Unable to retrieve session status'
            ], 500);
        }
    }

    /**
     * Get conversation history for a session
     */
    public function getConversationHistory(Request $request): JsonResponse
    {
        $validator = Validator::make($request->all(), [
            'session_id' => 'required|string|max:255',
            'limit' => 'nullable|integer|min:1|max:100'
        ]);

        if ($validator->fails()) {
            return response()->json([
                'success' => false,
                'error' => 'Invalid request parameters'
            ], 422);
        }

        try {
            $limit = $request->input('limit', 20);
            $history = $this->auraBotService->getConversationHistory(
                $request->input('session_id'),
                $limit
            );

            return response()->json([
                'success' => true,
                'messages' => $history,
                'total_count' => count($history)
            ]);

        } catch (\Exception $e) {
            Log::error('Conversation history error', [
                'error' => $e->getMessage(),
                'session_id' => $request->input('session_id')
            ]);

            return response()->json([
                'success' => false,
                'error' => 'Unable to retrieve conversation history'
            ], 500);
        }
    }

    /**
     * Reset session attempts (admin only)
     */
    public function resetSession(Request $request): JsonResponse
    {
        $validator = Validator::make($request->all(), [
            'session_id' => 'required|string|max:255'
        ]);

        if ($validator->fails()) {
            return response()->json([
                'success' => false,
                'error' => 'Invalid session ID'
            ], 422);
        }

        try {
            $success = $this->auraBotService->resetSession($request->input('session_id'));

            return response()->json([
                'success' => $success,
                'message' => $success ? 'Session reset successfully' : 'Session not found'
            ]);

        } catch (\Exception $e) {
            Log::error('Session reset error', [
                'error' => $e->getMessage(),
                'session_id' => $request->input('session_id')
            ]);

            return response()->json([
                'success' => false,
                'error' => 'Unable to reset session'
            ], 500);
        }
    }

    /**
     * Health check endpoint for AuraBot service
     */
    public function healthCheck(): JsonResponse
    {
        try {
            // Test database connection
            $dbStatus = \DB::connection()->getPdo() ? 'connected' : 'disconnected';

            // Test Nebius API connection
            $nebiusClient = app(\App\Services\NebiusClient::class);
            $apiTest = $nebiusClient->testConnection();

            return response()->json([
                'success' => true,
                'status' => 'healthy',
                'database' => $dbStatus,
                'nebius_api' => $apiTest['success'] ? 'connected' : 'error',
                'nebius_error' => $apiTest['error'] ?? null,
                'timestamp' => now()->toISOString()
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'status' => 'unhealthy',
                'error' => $e->getMessage(),
                'timestamp' => now()->toISOString()
            ], 500);
        }
    }
}

