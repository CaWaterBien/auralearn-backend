<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\DB;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('rag_documents', function (Blueprint $table) {
            $table->id();
            $table->string('source')->nullable()->comment('Source file, URL, or identifier');
            $table->string('document_type')->default('text')->comment('Type: text, html, css, lesson, activity');
            $table->text('content')->comment('Original document content');
            $table->text('chunk_text')->comment('Chunked text for embedding');
            $table->integer('chunk_index')->default(0)->comment('Chunk number in document');
            $table->json('metadata')->nullable()->comment('Additional metadata like course_id, lesson_id, etc');
            $table->json('embedding')->comment('Vector embedding as JSON array');
            $table->integer('embedding_dimensions')->default(1024)->comment('Dimensions of the embedding vector');
            $table->string('embedding_model')->default('BAAI/bge-multilingual-gemma2')->comment('Model used for embedding');
            $table->timestamps();

            // Indexes for efficient querying
            $table->index(['source', 'chunk_index']);
            $table->index('document_type');
            $table->index('created_at');
        });

        // Add pgvector extension if available (optional - graceful fallback)
        // Note: The system works perfectly without pgvector using JSON-based similarity
        if (DB::connection()->getDriverName() === 'pgsql') {
            try {
                // Check if pgvector is available before trying to create extension
                $extensionExists = DB::select("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'");
                
                if ($extensionExists) {
                    DB::statement('CREATE EXTENSION IF NOT EXISTS vector');
                    
                    // Add vector column for efficient similarity search
                    DB::statement('ALTER TABLE rag_documents ADD COLUMN embedding_vector vector(' . (env('VECTOR_DIM', 1024)) . ')');
                    
                    // Create index for vector similarity search  
                    DB::statement('CREATE INDEX rag_documents_embedding_vector_idx ON rag_documents USING ivfflat (embedding_vector vector_cosine_ops)');
                    
                    echo "✅ pgvector extension enabled for optimized vector search\n";
                } else {
                    echo "ℹ️  pgvector extension not available - using JSON fallback (fully functional)\n";
                }
            } catch (\Exception $e) {
                echo "ℹ️  pgvector extension not available - using JSON fallback (fully functional)\n";
                echo "   System will use cosine similarity calculation in PHP\n";
                // Continue without pgvector - system will use JSON-based similarity
            }
        }
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('rag_documents');
    }
};
