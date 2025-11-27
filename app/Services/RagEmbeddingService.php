<?php

namespace App\Services;

use App\Models\RagDocument;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Log;

class RagEmbeddingService
{
    private NebiusClient $nebiusClient;

    public function __construct(NebiusClient $nebiusClient)
    {
        $this->nebiusClient = $nebiusClient;
    }

    /**
     * Generate embedding for text with caching
     */
    public function generateEmbedding(string $text, string $model = 'BAAI/bge-multilingual-gemma2'): array
    {
        $textHash = hash('sha256', $text);

        // Check cache first
        $cached = DB::table('rag_embeddings_cache')
            ->where('text_hash', $textHash)
            ->where('model', $model)
            ->first();

        if ($cached) {
            // Update usage statistics
            DB::table('rag_embeddings_cache')
                ->where('id', $cached->id)
                ->update([
                    'usage_count' => $cached->usage_count + 1,
                    'last_used_at' => now()
                ]);

            Log::info('Embedding cache hit', ['text_length' => strlen($text)]);
            return json_decode($cached->embedding, true);
        }

        // Generate new embedding
        try {
            $embedding = $this->nebiusClient->createEmbedding($text, $model);

            // Cache the embedding
            DB::table('rag_embeddings_cache')->insert([
                'text_hash' => $textHash,
                'original_text' => $text,
                'embedding' => json_encode($embedding),
                'model' => $model,
                'dimensions' => count($embedding),
                'usage_count' => 1,
                'last_used_at' => now(),
                'created_at' => now(),
                'updated_at' => now()
            ]);

            Log::info('New embedding generated and cached', [
                'text_length' => strlen($text),
                'dimensions' => count($embedding)
            ]);

            return $embedding;

        } catch (\Exception $e) {
            Log::error('Embedding generation failed', [
                'error' => $e->getMessage(),
                'text_length' => strlen($text)
            ]);
            throw $e;
        }
    }

    /**
     * Split text into chunks for embedding
     */
    public function chunkText(string $text, int $chunkSize = null, int $overlap = null): array
    {
        $chunkSize = $chunkSize ?? env('RAG_CHUNK_SIZE', 1000);
        $overlap = $overlap ?? env('RAG_CHUNK_OVERLAP', 200);

        // Clean and normalize text
        $text = trim(preg_replace('/\s+/', ' ', $text));
        
        if (strlen($text) <= $chunkSize) {
            return [$text];
        }

        $chunks = [];
        $start = 0;

        while ($start < strlen($text)) {
            $end = min($start + $chunkSize, strlen($text));
            
            // Try to break at sentence boundaries
            if ($end < strlen($text)) {
                $lastPeriod = strrpos(substr($text, $start, $chunkSize), '.');
                $lastNewline = strrpos(substr($text, $start, $chunkSize), "\n");
                $lastSpace = strrpos(substr($text, $start, $chunkSize), ' ');
                
                $breakPoint = max($lastPeriod, $lastNewline, $lastSpace);
                if ($breakPoint !== false && $breakPoint > $chunkSize * 0.7) {
                    $end = $start + $breakPoint + 1;
                }
            }

            $chunk = trim(substr($text, $start, $end - $start));
            if (!empty($chunk)) {
                $chunks[] = $chunk;
            }

            // Move start position with overlap
            $start = max($start + $chunkSize - $overlap, $end);
        }

        return $chunks;
    }

    /**
     * Ingest document and create embeddings
     */
    public function ingestDocument(
        string $content,
        string $source,
        string $documentType = 'text',
        array $metadata = []
    ): int {
        $chunks = $this->chunkText($content);
        $insertedCount = 0;

        foreach ($chunks as $index => $chunk) {
            try {
                $embedding = $this->generateEmbedding($chunk);

                $document = RagDocument::create([
                    'source' => $source,
                    'document_type' => $documentType,
                    'content' => $content,
                    'chunk_text' => $chunk,
                    'chunk_index' => $index,
                    'metadata' => array_merge($metadata, [
                        'chunk_size' => strlen($chunk),
                        'total_chunks' => count($chunks)
                    ]),
                    'embedding' => $embedding,
                    'embedding_dimensions' => count($embedding),
                    'embedding_model' => env('EMBEDDING_MODEL', 'BAAI/bge-multilingual-gemma2')
                ]);

                // Update vector column if using pgvector
                if (DB::connection()->getDriverName() === 'pgsql') {
                    $vectorString = '[' . implode(',', $embedding) . ']';
                    DB::statement(
                        'UPDATE rag_documents SET embedding_vector = ?::vector WHERE id = ?',
                        [$vectorString, $document->id]
                    );
                }

                $insertedCount++;

                Log::info('Document chunk ingested', [
                    'source' => $source,
                    'chunk_index' => $index,
                    'chunk_size' => strlen($chunk)
                ]);

            } catch (\Exception $e) {
                Log::error('Failed to ingest chunk', [
                    'source' => $source,
                    'chunk_index' => $index,
                    'error' => $e->getMessage()
                ]);
                // Continue with next chunk
            }
        }

        return $insertedCount;
    }

    /**
     * Search for relevant documents
     */
    public function searchRelevantDocuments(
        string $query,
        int $limit = null,
        float $threshold = 0.7,
        array $documentTypes = []
    ): \Illuminate\Support\Collection {
        $limit = $limit ?? env('RAG_MAX_CHUNKS', 5);

        // Generate embedding for query
        $queryEmbedding = $this->generateEmbedding($query, env('EMBEDDING_MODEL', 'BAAI/bge-multilingual-gemma2'));

        // Create base query
        $documentsQuery = RagDocument::query();

        // Filter by document types if specified
        if (!empty($documentTypes)) {
            $documentsQuery->whereIn('document_type', $documentTypes);
        }

        // Find similar documents
        $results = RagDocument::findSimilar($queryEmbedding, $limit, $threshold);

        Log::info('RAG search completed', [
            'query_length' => strlen($query),
            'results_count' => $results->count(),
            'threshold' => $threshold
        ]);

        return $results;
    }

    /**
     * Clean up old cache entries
     */
    public function cleanupEmbeddingCache(int $daysOld = 30): int
    {
        $deleted = DB::table('rag_embeddings_cache')
            ->where('last_used_at', '<', now()->subDays($daysOld))
            ->delete();

        Log::info('Embedding cache cleanup', ['deleted_entries' => $deleted]);

        return $deleted;
    }
}

