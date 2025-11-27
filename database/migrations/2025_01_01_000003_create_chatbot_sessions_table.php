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
        Schema::create('chatbot_sessions', function (Blueprint $table) {
            $table->id();
            $table->string('session_id')->unique()->comment('Frontend session identifier');
            $table->unsignedBigInteger('user_id')->nullable()->comment('Authenticated user ID');
            $table->integer('attempt_count')->default(0)->comment('Number of questions asked in this session');
            $table->integer('max_attempts')->default(3)->comment('Maximum attempts allowed per session');
            $table->boolean('is_blocked')->default(false)->comment('Whether session is blocked from asking questions');
            $table->timestamp('last_activity')->nullable()->comment('Last interaction time');
            $table->timestamp('blocked_until')->nullable()->comment('When session will be unblocked');
            $table->json('progress_data')->nullable()->comment('User progress tracking data');
            $table->timestamps();

            // Indexes for efficient querying
            $table->index('session_id');
            $table->index(['user_id', 'last_activity']);
            $table->index(['is_blocked', 'blocked_until']);
            
            // Foreign key constraint
            $table->foreign('user_id')->references('id')->on('users')->onDelete('set null');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('chatbot_sessions');
    }
};

