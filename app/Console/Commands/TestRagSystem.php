<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Services\NebiusClient;
use App\Services\RagEmbeddingService;
use App\Services\AuraBotRagService;
use App\Models\RagDocument;
use Illuminate\Support\Str;

class TestRagSystem extends Command
{
    protected $signature = 'rag:test
                            {--question=What is HTML? : Test question to ask}
                            {--session-id=test_session : Session ID for testing}';

    protected $description = 'Test the complete RAG system functionality';

    public function handle(): int
    {
        $this->info('🤖 Testing AuraLearn RAG System...');
        $this->newLine();

        $question = $this->option('question');
        $sessionId = $this->option('session-id');

        // Test 1: Database connection
        $this->info('1. Testing database connection...');
        try {
            $docCount = RagDocument::count();
            $this->info("✅ Database connected. Found {$docCount} RAG documents.");
        } catch (\Exception $e) {
            $this->error("❌ Database connection failed: " . $e->getMessage());
            return 1;
        }

        // Test 2: Nebius API connection
        $this->info('2. Testing Nebius API connection...');
        try {
            $nebiusClient = app(NebiusClient::class);
            $result = $nebiusClient->testConnection();
            
            if ($result['success']) {
                $this->info("✅ Nebius API connected. Model: " . ($result['model'] ?? 'Unknown'));
                $this->info("   Response: " . $result['response']);
            } else {
                $this->error("❌ Nebius API connection failed: " . $result['error']);
                return 1;
            }
        } catch (\Exception $e) {
            $this->error("❌ Nebius API test failed: " . $e->getMessage());
            return 1;
        }

        // Test 3: Embedding generation
        $this->info('3. Testing embedding generation...');
        try {
            $embeddingService = app(RagEmbeddingService::class);
            $embedding = $embeddingService->generateEmbedding("Test embedding text");
            $this->info("✅ Embedding generated. Dimensions: " . count($embedding));
        } catch (\Exception $e) {
            $this->error("❌ Embedding generation failed: " . $e->getMessage());
            return 1;
        }

        // Test 4: RAG search
        $this->info('4. Testing RAG document search...');
        try {
            $results = $embeddingService->searchRelevantDocuments($question, 3);
            $this->info("✅ RAG search completed. Found {$results->count()} relevant documents.");
            
            if ($results->count() > 0) {
                $this->info("   Top result: " . $results->first()->source);
                $this->info("   Similarity: " . round(($results->first()->similarity_score ?? 0) * 100, 1) . "%");
            }
        } catch (\Exception $e) {
            $this->error("❌ RAG search failed: " . $e->getMessage());
            return 1;
        }

        // Test 5: Complete AuraBot workflow
        $this->info('5. Testing complete AuraBot workflow...');
        try {
            $auraBotService = app(AuraBotRagService::class);
            $response = $auraBotService->processUserQuestion($sessionId, $question);
            
            if ($response['success']) {
                $this->info("✅ AuraBot workflow completed successfully!");
                $this->info("   Response length: " . strlen($response['response']));
                $this->info("   Tokens used: " . ($response['tokens_used'] ?? 'Unknown'));
                $this->info("   Remaining attempts: " . $response['remaining_attempts']);
                $this->newLine();
                $this->info("🤖 AuraBot Response:");
                $this->info("   " . Str::limit($response['response'], 200));
            } else {
                $this->error("❌ AuraBot workflow failed: " . $response['error']);
                return 1;
            }
        } catch (\Exception $e) {
            $this->error("❌ AuraBot workflow test failed: " . $e->getMessage());
            return 1;
        }

        $this->newLine();
        $this->info('🎉 All tests passed! RAG system is fully functional.');
        $this->newLine();

        // Additional info
        $this->info('📊 System Statistics:');
        $this->info("   Total RAG documents: " . RagDocument::count());
        $this->info("   Document types: " . RagDocument::distinct('document_type')->pluck('document_type')->implode(', '));
        $this->info("   Cache entries: " . \DB::table('rag_embeddings_cache')->count());

        return 0;
    }
}
