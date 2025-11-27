<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('rag_embeddings_cache', function (Blueprint $table) {
            $table->id();
            $table->string('text_hash', 64)->unique()->comment('SHA256 hash of the text');
            $table->text('original_text')->comment('Original text that was embedded');
            $table->json('embedding')->comment('Cached embedding vector');
            $table->string('model')->default('BAAI/bge-multilingual-gemma2')->comment('Embedding model used');
            $table->integer('dimensions')->default(1024)->comment('Vector dimensions');
            $table->integer('usage_count')->default(1)->comment('How many times this embedding was reused');
            $table->timestamp('last_used_at')->nullable()->comment('When this cache was last accessed');
            $table->timestamps();

            // Indexes for efficient querying
            $table->index('text_hash');
            $table->index(['model', 'last_used_at']);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('rag_embeddings_cache');
    }
};

