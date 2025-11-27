<?php

namespace App\Services;

use App\Models\ChatbotConversation;
use App\Models\ChatbotSession;
use App\Models\RagDocument;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Str;

class AuraBotRagService
{
    private NebiusClient $nebiusClient;
    private RagEmbeddingService $embeddingService;

    public function __construct(NebiusClient $nebiusClient, RagEmbeddingService $embeddingService)
    {
        $this->nebiusClient = $nebiusClient;
        $this->embeddingService = $embeddingService;
    }

    /**
     * Process user question and generate RAG-powered response
     */
    public function processUserQuestion(
        string $sessionId,
        string $question,
        ?string $htmlContext = null,
        ?string $instructionsContext = null,
        ?string $feedbackContext = null,
        ?int $userId = null
    ): array {
        
        // Get or create session
        $session = ChatbotSession::getOrCreate($sessionId, $userId);

        // UNLIMITED QUESTIONS: Removed question limit check per user request
        // Users can now ask unlimited questions to AuraBot

        try {
            // Generate unique message ID
            $userMessageId = Str::uuid();
            $assistantMessageId = Str::uuid();

            // Save user message
            ChatbotConversation::saveUserMessage(
                $sessionId,
                $userMessageId,
                $question,
                $userId,
                $htmlContext,
                $instructionsContext,
                ['attempt_number' => $session->attempt_count + 1]
            );

            // Get conversation history
            $conversationHistory = ChatbotConversation::getRecentContext($sessionId, 5);

            // Search for relevant documents
            $relevantDocs = $this->embeddingService->searchRelevantDocuments(
                $question,
                env('RAG_MAX_CHUNKS', 5),
                0.7,
                ['html', 'lesson', 'activity', 'tutorial']
            );

            // Build context from retrieved documents
            $retrievedContext = $this->buildRetrievedContext($relevantDocs);

            // Build conversation context
            $conversationContext = $this->buildConversationContext($conversationHistory);

            // Build HTML/Instructions/Feedback context
            $editorContext = $this->buildEditorContext($htmlContext, $instructionsContext, $feedbackContext);

            // Generate AI response
            $aiResponse = $this->generateAuraBotResponse(
                $question,
                $retrievedContext,
                $conversationContext,
                $editorContext,
                $session->attempt_count + 1
            );

            // Save assistant message
            ChatbotConversation::saveAssistantMessage(
                $sessionId,
                $assistantMessageId,
                $aiResponse['content'],
                $userId,
                $relevantDocs->map(function ($doc) {
                    return [
                        'id' => $doc->id,
                        'source' => $doc->source,
                        'similarity_score' => $doc->similarity_score ?? 0,
                        'chunk_text' => substr($doc->chunk_text, 0, 200) . '...'
                    ];
                })->toArray(),
                $aiResponse['tokens_used']
            );

            // UNLIMITED QUESTIONS: Removed attempt increment
            // $session->incrementAttempt(); // Commented out for unlimited questions

            // Update progress data
            $this->updateUserProgress($session, $question, $aiResponse['content']);

            return [
                'success' => true,
                'response' => $aiResponse['content'],
                'message_id' => $assistantMessageId,
                'remaining_attempts' => 999, // Unlimited questions
                'tokens_used' => $aiResponse['tokens_used'],
                'retrieved_sources' => $relevantDocs->pluck('source')->unique()->values()->toArray(),
                'session_info' => [
                    'attempt_count' => $session->attempt_count,
                    'max_attempts' => 999, // Unlimited questions
                    'is_blocked' => false // Never blocked
                ]
            ];

        } catch (\Exception $e) {
            Log::error('AuraBot processing error', [
                'session_id' => $sessionId,
                'error' => $e->getMessage(),
                'question_length' => strlen($question)
            ]);

            return [
                'success' => false,
                'error' => 'I apologize, but I encountered an error processing your question. Please try again.',
                'remaining_attempts' => $session->getRemainingAttempts()
            ];
        }
    }

    /**
     * Build context from retrieved RAG documents
     */
    private function buildRetrievedContext(\Illuminate\Support\Collection $documents): string
    {
        if ($documents->isEmpty()) {
            return "No specific relevant content found in the knowledge base.";
        }

        $context = "Relevant learning material:\n\n";
        
        foreach ($documents as $doc) {
            $context .= "Source: {$doc->source}\n";
            $context .= "Content: " . trim($doc->chunk_text) . "\n";
            if ($doc->similarity_score ?? false) {
                $context .= "Relevance: " . round($doc->similarity_score * 100, 1) . "%\n";
            }
            $context .= "---\n";
        }

        return $context;
    }

    /**
     * Build conversation context from history
     */
    private function buildConversationContext(\Illuminate\Support\Collection $history): string
    {
        if ($history->isEmpty()) {
            return "This is the start of our conversation.";
        }

        $context = "Previous conversation context:\n\n";
        
        foreach ($history as $message) {
            $role = $message->role === 'user' ? 'Student' : 'AuraBot';
            $content = Str::limit($message->content, 200);
            $context .= "{$role}: {$content}\n";
        }

        return $context;
    }

    /**
     * Build editor context from HTML, instructions, and feedback
     */
    private function buildEditorContext(?string $htmlContext, ?string $instructionsContext, ?string $feedbackContext): string
    {
        $context = "";

        if ($htmlContext) {
            $context .= "Current HTML code in editor:\n```html\n" . trim($htmlContext) . "\n```\n\n";
        }

        if ($instructionsContext) {
            $context .= "Activity instructions:\n" . trim($instructionsContext) . "\n\n";
        }

        if ($feedbackContext) {
            $context .= "Previous submission feedback:\n" . trim($feedbackContext) . "\n\n";
        }

        if (empty($context)) {
            $context = "No current code, instructions, or feedback context available.";
        }

        return $context;
    }

    /**
     * Generate AuraBot response using Nebius API
     */
    private function generateAuraBotResponse(
        string $question,
        string $retrievedContext,
        string $conversationContext,
        string $editorContext,
        int $attemptNumber
    ): array {
        
        $systemPrompt = $this->buildSystemPrompt($attemptNumber);
        $userPrompt = $this->buildUserPrompt($question, $retrievedContext, $conversationContext, $editorContext);

        $messages = [
            ['role' => 'system', 'content' => $systemPrompt],
            ['role' => 'user', 'content' => $userPrompt]
        ];

        $response = $this->nebiusClient->createChatCompletion($messages, [
            'max_tokens' => env('AURABOT_MAX_TOKENS', 5000),
            'temperature' => 0.1
        ]);

        $content = $response['choices'][0]['message']['content'] ?? 'I apologize, but I could not generate a response.';
        $tokensUsed = $response['usage']['total_tokens'] ?? 0;

        return [
            'content' => $content,
            'tokens_used' => $tokensUsed
        ];
    }

    /**
     * Build system prompt for AuraBot
     */
    private function buildSystemPrompt(int $attemptNumber): string
    {
        return "You are AuraBot, a friendly and interactive HTML learning assistant! 🤖

PERSONALITY & APPROACH:
- Be conversational, encouraging, and enthusiastic about HTML
- Use emojis strategically to make responses engaging
- Never provide complete code solutions - that ruins the learning!
- Focus ONLY on HTML fundamentals (no CSS for now)

CORE MISSION:
1. 🎯 FIRST: Read and understand what the user is actually asking
2. 🚫 NEVER write code for students - guide them to discover it
3. 💬 Respond appropriately to their specific question or request
4. 🔍 Only analyze code when they ask about their code specifically
5. 💡 Give hints and guidance based on their actual needs
6. 🤔 Ask guiding questions to make them think

RESPONSE TYPES (respond based on what they're asking):
- If they ask about their code → Analyze and give specific feedback
- If they ask for help finishing → Focus on what's missing for completion
- If they ask general HTML questions → Teach the concept with hints
- If they ask for code → Politely refuse and guide them instead
- If they're stuck → Encourage and give small next steps
- If they greet you → Be friendly and ask how you can help

RESPONSE GUIDELINES:
- Start by acknowledging their specific question
- Tailor your response to what they actually asked
- Don't default to code analysis unless they ask about their code
- Keep responses friendly, concise (under 300 words), and actionable
- End with encouragement or a guiding question

HTML FOCUS AREAS:
- Document structure (DOCTYPE, html, head, body)
- Semantic elements (header, main, section, footer)
- Content elements (h1-h6, p, img, lists, tables, forms)
- Accessibility (alt attributes, labels, proper nesting)

Remember: Understand their question FIRST, then respond appropriately! 🚀";
    }

    /**
     * Build user prompt with all context
     */
    private function buildUserPrompt(
        string $question,
        string $retrievedContext,
        string $conversationContext,
        string $editorContext
    ): string {
        
        // Analyze the HTML code to provide better context
        $htmlAnalysis = $this->analyzeStudentHtml($editorContext);
        
        $prompt = "🎯 STUDENT QUESTION: \"{$question}\"

💬 RECENT CONVERSATION:
{$conversationContext}

📚 RELEVANT LEARNING MATERIALS:
{$retrievedContext}

📝 THEIR CURRENT HTML CODE (for reference if needed):
{$htmlAnalysis}

🎯 YOUR TASK:
1. FIRST: Read and understand their specific question
2. Respond directly to what they're asking
3. Use their code context ONLY if relevant to their question
4. Focus on HTML fundamentals and learning
5. Be encouraging and provide appropriate guidance

RESPONSE APPROACH:
- If asking about their code → Analyze and give specific feedback
- If asking for help finishing → Guide them toward completion
- If asking general questions → Teach the concept
- If asking for code → Refuse politely and guide instead
- If stuck or confused → Encourage and give small steps

Remember: UNDERSTAND their question FIRST, then respond appropriately! 🚀";

        return $prompt;
    }

    /**
     * Analyze student's HTML code to provide contextual feedback
     */
    private function analyzeStudentHtml(string $editorContext): string
    {
        // Look for the HTML code in the editor context
        if (empty($editorContext)) {
            return "⚠️ No editor context received.";
        }
        
        // Extract HTML code from the formatted context
        if (preg_match('/```html\s*(.*?)\s*```/s', $editorContext, $matches)) {
            $htmlCode = trim($matches[1] ?? '');
        } elseif (preg_match('/Current HTML code in editor:\s*(.*?)(?:Current instructions|$)/s', $editorContext, $matches)) {
            $htmlCode = trim($matches[1] ?? '');
        } else {
            return "⚠️ No HTML code detected in the editor context.";
        }
        
        if (empty($htmlCode)) {
            return "📝 The editor appears to be empty. Student needs to start with basic HTML structure.";
        }
        
        $analysis = [];
        
        // Check for DOCTYPE
        if (stripos($htmlCode, '<!DOCTYPE html>') !== false) {
            $analysis[] = "✅ Has DOCTYPE declaration (good start!)";
        } else {
            $analysis[] = "❌ Missing DOCTYPE declaration";
        }
        
        // Check for HTML tag
        if (preg_match('/<html[^>]*>/i', $htmlCode)) {
            $analysis[] = "✅ Has <html> tag";
            
            // Check for lang attribute
            if (preg_match('/<html[^>]*lang=["\'][^"\']*["\'][^>]*>/i', $htmlCode)) {
                $analysis[] = "✅ <html> has lang attribute (accessibility!)";
            } else {
                $analysis[] = "💡 <html> missing lang attribute (accessibility)";
            }
        } else {
            $analysis[] = "❌ Missing <html> tag";
        }
        
        // Check for head section
        if (preg_match('/<head[^>]*>.*?<\/head>/is', $htmlCode)) {
            $analysis[] = "✅ Has <head> section";
            
            // Check for title
            if (preg_match('/<title[^>]*>(.*?)<\/title>/is', $htmlCode, $titleMatch)) {
                $titleText = trim(strip_tags($titleMatch[1]));
                if (!empty($titleText)) {
                    $analysis[] = "✅ Has <title>: \"{$titleText}\"";
                } else {
                    $analysis[] = "💡 <title> tag exists but is empty";
                }
            } else {
                $analysis[] = "❌ Missing <title> tag in <head>";
            }
        } else {
            $analysis[] = "❌ Missing <head> section";
        }
        
        // Check for body section
        if (preg_match('/<body[^>]*>(.*?)<\/body>/is', $htmlCode, $bodyMatch)) {
            $analysis[] = "✅ Has <body> section";
            
            $bodyContent = trim($bodyMatch[1]);
            
            // Analyze body content
            if (empty($bodyContent) || strip_tags($bodyContent) === '' || preg_match('/^\s*<!--.*?-->\s*$/s', $bodyContent)) {
                $analysis[] = "📝 <body> is empty or only contains comments";
            } else {
                // Check for headings
                if (preg_match('/<h[1-6][^>]*>/i', $bodyContent)) {
                    $analysis[] = "✅ Has heading elements";
                } else {
                    $analysis[] = "💡 No heading elements found";
                }
                
                // Check for paragraphs
                if (preg_match('/<p[^>]*>/i', $bodyContent)) {
                    $analysis[] = "✅ Has paragraph elements";
                } else {
                    $analysis[] = "💡 No paragraph elements found";
                }
                
                // Check for images
                if (preg_match('/<img[^>]*>/i', $bodyContent)) {
                    if (preg_match('/<img[^>]*alt=["\'][^"\']*["\'][^>]*>/i', $bodyContent)) {
                        $analysis[] = "✅ Has image with alt text";
                    } else {
                        $analysis[] = "⚠️ Has image but missing alt attribute";
                    }
                } else {
                    $analysis[] = "💡 No images found";
                }
                
                // Check for lists
                if (preg_match('/<[uo]l[^>]*>/i', $bodyContent)) {
                    $analysis[] = "✅ Has list elements";
                } else {
                    $analysis[] = "💡 No list elements found";
                }
                
                // Check for tables
                if (preg_match('/<table[^>]*>/i', $bodyContent)) {
                    $analysis[] = "✅ Has table element";
                } else {
                    $analysis[] = "💡 No table elements found";
                }
                
                // Check for forms
                if (preg_match('/<form[^>]*>/i', $bodyContent)) {
                    $analysis[] = "✅ Has form element";
                } else {
                    $analysis[] = "💡 No form elements found";
                }
                
                // Check for semantic elements
                $semanticElements = ['header', 'nav', 'main', 'section', 'article', 'aside', 'footer'];
                $foundSemantic = [];
                foreach ($semanticElements as $element) {
                    if (preg_match("/<{$element}[^>]*>/i", $bodyContent)) {
                        $foundSemantic[] = $element;
                    }
                }
                
                if (!empty($foundSemantic)) {
                    $analysis[] = "✅ Has semantic elements: " . implode(', ', $foundSemantic);
                } else {
                    $analysis[] = "💡 No semantic HTML5 elements found";
                }
            }
        } else {
            $analysis[] = "❌ Missing <body> section";
        }
        
        // Check for common issues
        if (preg_match('/<br\s*\/?>/i', $htmlCode)) {
            $analysis[] = "⚠️ Uses <br> tags (consider paragraph structure)";
        }
        
        return "STRUCTURE ANALYSIS:\n" . implode("\n", $analysis) . "\n\nCODE LENGTH: " . strlen($htmlCode) . " characters";
    }

    /**
     * Update user progress based on interaction
     */
    private function updateUserProgress(ChatbotSession $session, string $question, string $response): void
    {
        $currentProgress = $session->progress_data ?? [];

        // Analyze question for topics
        $topics = $this->extractTopicsFromQuestion($question);
        
        // Update progress data
        $updatedProgress = array_merge($currentProgress, [
            'total_questions' => ($currentProgress['total_questions'] ?? 0) + 1,
            'topics_discussed' => array_unique(array_merge($currentProgress['topics_discussed'] ?? [], $topics)),
            'last_question_topics' => $topics,
            'last_interaction' => now()->toISOString(),
            'question_history' => array_slice(
                array_merge($currentProgress['question_history'] ?? [], [$question]),
                -10 // Keep last 10 questions
            )
        ]);

        $session->updateProgress($updatedProgress);
    }

    /**
     * Extract topics from user question
     */
    private function extractTopicsFromQuestion(string $question): array
    {
        $lowerQuestion = strtolower($question);
        $topics = [];

        $topicMap = [
            'html' => ['html', 'element', 'tag', 'markup', 'structure'],
            'css' => ['css', 'style', 'styling', 'color', 'font', 'layout'],
            'flexbox' => ['flexbox', 'flex', 'flexible'],
            'grid' => ['grid', 'css grid', 'grid layout'],
            'responsive' => ['responsive', 'mobile', 'media query', 'breakpoint'],
            'semantic' => ['semantic', 'accessibility', 'aria', 'alt'],
            'forms' => ['form', 'input', 'button', 'textarea', 'select'],
            'javascript' => ['javascript', 'js', 'script', 'dom'],
            'debugging' => ['error', 'bug', 'fix', 'debug', 'problem', 'issue'],
            'best_practices' => ['best practice', 'convention', 'standard', 'clean code']
        ];

        foreach ($topicMap as $topic => $keywords) {
            foreach ($keywords as $keyword) {
                if (strpos($lowerQuestion, $keyword) !== false) {
                    $topics[] = $topic;
                    break;
                }
            }
        }

        return array_unique($topics);
    }

    /**
     * Get session status and info
     */
    public function getSessionStatus(string $sessionId): array
    {
        $session = ChatbotSession::where('session_id', $sessionId)->first();

        if (!$session) {
            return [
                'exists' => false,
                'can_ask' => true,
                'remaining_attempts' => 999, // Unlimited questions
                'attempt_count' => 0
            ];
        }

        return [
            'exists' => true,
            'can_ask' => true, // Always true for unlimited questions
            'remaining_attempts' => 999, // Unlimited questions
            'attempt_count' => $session->attempt_count,
            'is_blocked' => false, // Never blocked for unlimited questions
            'blocked_until' => null,
            'progress' => $session->progress_data ?? []
        ];
    }

    /**
     * Get conversation history for frontend
     */
    public function getConversationHistory(string $sessionId, int $limit = 20): array
    {
        $messages = ChatbotConversation::getSessionHistory($sessionId, $limit);

        return $messages->map(function ($message) {
            return [
                'id' => $message->message_id,
                'role' => $message->role,
                'content' => $message->content,
                'timestamp' => $message->sent_at->toISOString(),
                'metadata' => $message->metadata ?? []
            ];
        })->toArray();
    }

    /**
     * Reset session (admin function)
     */
    public function resetSession(string $sessionId): bool
    {
        $session = ChatbotSession::where('session_id', $sessionId)->first();
        
        if ($session) {
            $session->resetAttempts();
            Log::info('Session reset', ['session_id' => $sessionId]);
            return true;
        }

        return false;
    }
}

