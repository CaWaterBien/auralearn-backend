<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Models\RagDocument;
use App\Models\ChatbotSession;
use App\Models\ChatbotConversation;

class TestSystemWithoutAPIs extends Command
{
    protected $signature = 'rag:test-offline';
    protected $description = 'Test RAG system functionality without external APIs';

    public function handle(): int
    {
        $this->info('🤖 Testing AuraBot RAG System (Offline Mode)...');
        $this->newLine();

        // Test 1: Database connections
        $this->info('1. Testing database tables...');
        try {
            $ragCount = RagDocument::count();
            $sessionCount = ChatbotSession::count();
            $conversationCount = ChatbotConversation::count();
            
            $this->info("✅ Database tables working:");
            $this->info("   - RAG Documents: {$ragCount}");
            $this->info("   - Sessions: {$sessionCount}");
            $this->info("   - Conversations: {$conversationCount}");
        } catch (\Exception $e) {
            $this->error("❌ Database test failed: " . $e->getMessage());
            return 1;
        }

        // Test 2: Create mock RAG data
        $this->info('2. Creating test RAG documents...');
        try {
            // Create sample documents without embeddings
            $sampleDocs = [
                [
                    'content' => 'HTML is the standard markup language for creating web pages. It uses elements like <h1>, <p>, and <div> to structure content.',
                    'source' => 'html_basics_test.txt',
                    'type' => 'html'
                ],
                [
                    'content' => 'CSS is used to style HTML elements. Use selectors like .class, #id, and element to target HTML elements and apply styles.',
                    'source' => 'css_basics_test.txt', 
                    'type' => 'css'
                ]
            ];

            foreach ($sampleDocs as $doc) {
                // Create mock embedding (1536 zeros for testing)
                $mockEmbedding = array_fill(0, 1536, 0.1);
                
                RagDocument::create([
                    'source' => $doc['source'],
                    'document_type' => $doc['type'],
                    'content' => $doc['content'],
                    'chunk_text' => $doc['content'],
                    'chunk_index' => 0,
                    'metadata' => ['test_mode' => true],
                    'embedding' => $mockEmbedding,
                    'embedding_dimensions' => 1536,
                    'embedding_model' => 'mock-for-testing'
                ]);
            }

            $this->info("✅ Created 2 test RAG documents");
        } catch (\Exception $e) {
            $this->error("❌ RAG document creation failed: " . $e->getMessage());
            return 1;
        }

        // Test 3: Session management
        $this->info('3. Testing session management...');
        try {
            // Clear any existing test session first
            \DB::table('chatbot_sessions')->where('session_id', 'test_session_123')->delete();
            
            // Use PostgreSQL-specific boolean syntax
            \DB::statement("
                INSERT INTO chatbot_sessions 
                (session_id, user_id, attempt_count, max_attempts, is_blocked, last_activity, progress_data, created_at, updated_at) 
                VALUES (?, ?, ?, ?, ?::boolean, ?, ?::json, ?, ?)
            ", [
                'test_session_123',
                null,
                0,
                3,
                'false', // PostgreSQL boolean string
                now(),
                '[]',
                now(),
                now()
            ]);
            
            $session = ChatbotSession::where('session_id', 'test_session_123')->first();
            $this->info("✅ Session created: ID {$session->id}");
            $this->info("   - Remaining attempts: {$session->getRemainingAttempts()}");
            $this->info("   - Can ask: " . ($session->canAskQuestion() ? 'Yes' : 'No'));
        } catch (\Exception $e) {
            $this->error("❌ Session test failed: " . $e->getMessage());
            return 1;
        }

        // Test 4: Conversation storage
        $this->info('4. Testing conversation storage...');
        try {
            $userMsg = ChatbotConversation::saveUserMessage(
                'test_session_123',
                'test_user_msg_' . time(),
                'How do I create a heading in HTML?',
                null,
                '<h1>Test</h1>',
                'Create a heading element'
            );

            $aiMsg = ChatbotConversation::saveAssistantMessage(
                'test_session_123',
                'test_ai_msg_' . time(),
                'Great question! In HTML, you can create headings using h1, h2, h3, h4, h5, or h6 elements. Try using <h1> for your main heading!',
                null,
                [['source' => 'html_basics_test.txt', 'similarity' => 0.95]],
                150
            );

            $this->info("✅ Conversation messages saved:");
            $this->info("   - User message ID: {$userMsg->id}");
            $this->info("   - AI message ID: {$aiMsg->id}");
        } catch (\Exception $e) {
            $this->error("❌ Conversation test failed: " . $e->getMessage());
            return 1;
        }

        // Test 5: Search functionality (with mock data)
        $this->info('5. Testing document search...');
        try {
            $query = [0.1, 0.2, 0.1]; // Simple mock query vector
            $results = RagDocument::findSimilar($query, 2, 0.5);
            $this->info("✅ Search completed. Found {$results->count()} results");
        } catch (\Exception $e) {
            $this->error("❌ Search test failed: " . $e->getMessage());
            return 1;
        }

        $this->newLine();
        $this->info('🎉 Offline tests passed! Core RAG system is functional.');
        $this->newLine();
        
        $this->info('📋 Next steps to complete setup:');
        $this->info('1. Add OPENAI_API_KEY to .env for embeddings');
        $this->info('2. Run: php artisan rag:create-samples (with API key)');
        $this->info('3. Test full system: php artisan rag:test');
        $this->info('4. Start servers: php artisan serve & npm run dev (in capstone-app)');

        return 0;
    }
}
