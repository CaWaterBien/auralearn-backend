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
        Schema::create('chatbot_conversations', function (Blueprint $table) {
            $table->id();
            $table->string('session_id')->comment('Frontend session identifier');
            $table->unsignedBigInteger('user_id')->nullable()->comment('Authenticated user ID');
            $table->string('message_id')->unique()->comment('Unique message identifier');
            $table->enum('role', ['user', 'assistant'])->comment('Message sender role');
            $table->text('content')->comment('Message content');
            $table->json('metadata')->nullable()->comment('Additional message metadata');
            $table->text('html_context')->nullable()->comment('HTML code from editor at time of message');
            $table->text('instructions_context')->nullable()->comment('Instructions/feedback context');
            $table->json('retrieved_chunks')->nullable()->comment('RAG chunks used for response');
            $table->integer('tokens_used')->nullable()->comment('Tokens used in AI response');
            $table->timestamp('sent_at')->comment('When message was sent');
            $table->timestamps();

            // Indexes for efficient querying
            $table->index(['session_id', 'sent_at']);
            $table->index(['user_id', 'sent_at']);
            $table->index('role');
            $table->index('sent_at');
            
            // Foreign key constraint
            $table->foreign('user_id')->references('id')->on('users')->onDelete('set null');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('chatbot_conversations');
    }
};

