<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Support\Facades\DB;

class RagDocument extends Model
{
    use HasFactory;

    protected $fillable = [
        'source',
        'document_type',
        'content',
        'chunk_text',
        'chunk_index',
        'metadata',
        'embedding',
        'embedding_dimensions',
        'embedding_model'
    ];

    protected $casts = [
        'metadata' => 'array',
        'embedding' => 'array',
        'chunk_index' => 'integer',
        'embedding_dimensions' => 'integer'
    ];

    /**
     * Find similar documents using vector similarity
     */
    public static function findSimilar(array $queryEmbedding, int $limit = 5, float $threshold = 0.7)
    {
        if (DB::connection()->getDriverName() === 'pgsql') {
            // Use pgvector for efficient similarity search
            $vectorString = '[' . implode(',', $queryEmbedding) . ']';
            
            return self::select([
                'rag_documents.*',
                DB::raw("1 - (embedding_vector <=> '{$vectorString}'::vector) as similarity_score")
            ])
            ->whereRaw("1 - (embedding_vector <=> '{$vectorString}'::vector) > ?", [$threshold])
            ->orderByRaw("embedding_vector <=> '{$vectorString}'::vector")
            ->limit($limit)
            ->get();
        } else {
            // Fallback to cosine similarity calculation in PHP
            $documents = self::all();
            $similarities = [];

            foreach ($documents as $doc) {
                $similarity = self::cosineSimilarity($queryEmbedding, $doc->embedding);
                if ($similarity > $threshold) {
                    $similarities[] = [
                        'document' => $doc,
                        'similarity_score' => $similarity
                    ];
                }
            }

            // Sort by similarity (highest first)
            usort($similarities, fn($a, $b) => $b['similarity_score'] <=> $a['similarity_score']);

            return collect(array_slice($similarities, 0, $limit))
                ->map(function ($item) {
                    $doc = $item['document'];
                    $doc->similarity_score = $item['similarity_score'];
                    return $doc;
                });
        }
    }

    /**
     * Calculate cosine similarity between two vectors
     */
    public static function cosineSimilarity(array $a, array $b): float
    {
        $dotProduct = 0.0;
        $normA = 0.0;
        $normB = 0.0;

        $count = min(count($a), count($b));
        
        for ($i = 0; $i < $count; $i++) {
            $dotProduct += $a[$i] * $b[$i];
            $normA += $a[$i] * $a[$i];
            $normB += $b[$i] * $b[$i];
        }

        if ($normA == 0 || $normB == 0) {
            return 0.0;
        }

        return $dotProduct / (sqrt($normA) * sqrt($normB));
    }

    /**
     * Get documents by type
     */
    public static function getByType(string $type)
    {
        return self::where('document_type', $type)->get();
    }

    /**
     * Get documents by source
     */
    public static function getBySource(string $source)
    {
        return self::where('source', $source)->orderBy('chunk_index')->get();
    }

    /**
     * Get document chunks with context (including neighboring chunks)
     */
    public function getChunksWithContext(int $contextSize = 1)
    {
        return self::where('source', $this->source)
            ->where('chunk_index', '>=', max(0, $this->chunk_index - $contextSize))
            ->where('chunk_index', '<=', $this->chunk_index + $contextSize)
            ->orderBy('chunk_index')
            ->get();
    }
}

