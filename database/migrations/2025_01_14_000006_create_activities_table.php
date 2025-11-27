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
        Schema::create('activities', function (Blueprint $table) {
            $table->id();
            $table->foreignId('lesson_id')->constrained()->onDelete('cascade');
            $table->string('title');
            $table->text('description')->nullable();
            $table->string('activity_type')->default('coding'); // Only coding activities are supported
            $table->longText('instructions'); // Activity instructions
            $table->json('questions')->nullable(); // For additional activity data
            $table->json('resources')->nullable(); // Links, files, etc.
            $table->integer('time_limit')->nullable(); // Time limit in minutes
            $table->integer('max_attempts')->nullable(); // Maximum attempts allowed
            $table->integer('passing_score')->nullable(); // Minimum score to pass (percentage)
            $table->integer('points')->default(0); // Points/score value
            $table->integer('order_index')->default(0);
            $table->integer('is_required')->default(0); // Is this activity required to complete the lesson
            $table->integer('is_published')->default(1);
            $table->json('metadata')->nullable(); // Additional activity-specific data
            $table->timestamps();
            
            $table->index(['lesson_id', 'order_index']);
            $table->index('activity_type');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('activities');
    }
};
