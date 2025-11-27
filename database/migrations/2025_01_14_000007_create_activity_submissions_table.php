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
        Schema::create('activity_submissions', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->onDelete('cascade');
            $table->foreignId('activity_id')->constrained()->onDelete('cascade');
            $table->longText('submitted_code'); // The code submitted by the user
            $table->longText('generated_output')->nullable(); // The actual HTML output generated
            $table->json('validation_results'); // Detailed validation results
            $table->json('instruction_compliance'); // Which instructions were followed
            $table->enum('completion_status', ['pending', 'passed', 'failed', 'needs_review']);
            $table->integer('attempt_number'); // Which attempt this is (1, 2, 3, etc.)
            $table->integer('score')->nullable(); // Score achieved (0-100)
            $table->integer('time_spent_minutes')->nullable(); // Time spent on this attempt
            $table->longText('feedback')->nullable(); // Automated feedback provided
            $table->json('error_details')->nullable(); // Specific errors found
            $table->boolean('is_completed')->default(false);
            $table->datetime('submitted_at');
            $table->datetime('completed_at')->nullable();
            $table->timestamps();
            
            // Indexes for performance
            $table->index(['user_id', 'activity_id']);
            $table->index(['activity_id', 'completion_status']);
            $table->index('completion_status');
            $table->index('submitted_at');
            
            // Ensure unique constraint per user per activity attempt
            $table->unique(['user_id', 'activity_id', 'attempt_number']);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('activity_submissions');
    }
};
