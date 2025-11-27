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
        Schema::create('activity_logs', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->nullable()->constrained()->onDelete('set null');
            $table->foreignId('activity_id')->constrained()->onDelete('cascade');
            $table->foreignId('submission_id')->nullable()->constrained('activity_submissions')->onDelete('cascade');
            $table->enum('event_type', [
                'activity_started', 
                'code_changed', 
                'validation_attempted', 
                'submission_created', 
                'hint_requested', 
                'activity_completed', 
                'activity_reset',
                'instructor_override',
                'admin_review'
            ]);
            $table->json('event_data'); // Store detailed event information
            $table->string('ip_address', 45)->nullable();
            $table->string('user_agent')->nullable();
            $table->timestamp('created_at');
            
            // Indexes for performance and analysis
            $table->index(['user_id', 'activity_id', 'created_at']);
            $table->index(['activity_id', 'event_type', 'created_at']);
            $table->index(['event_type', 'created_at']);
            $table->index('created_at');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('activity_logs');
    }
};
