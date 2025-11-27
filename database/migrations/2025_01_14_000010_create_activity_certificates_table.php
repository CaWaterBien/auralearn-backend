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
        Schema::create('activity_certificates', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->onDelete('cascade');
            $table->foreignId('activity_id')->constrained()->onDelete('cascade');
            $table->foreignId('submission_id')->constrained('activity_submissions')->onDelete('cascade');
            $table->string('certificate_id')->unique(); // Unique certificate identifier
            $table->enum('certificate_type', ['completion', 'excellence', 'first_attempt', 'perfect_score']);
            $table->string('badge_level', 20)->default('bronze'); // bronze, silver, gold, platinum
            $table->json('achievement_data'); // Score, attempts, time, special achievements
            $table->string('certificate_url')->nullable(); // Generated certificate image/PDF
            $table->boolean('is_verified')->default(true); // All certificates are verified by default
            $table->timestamp('earned_at');
            $table->timestamps();
            
            // Indexes
            $table->index(['user_id', 'activity_id']);
            $table->index(['certificate_type', 'badge_level']);
            $table->index('earned_at');
            
            // Unique constraint to prevent duplicate certificates
            $table->unique(['user_id', 'activity_id', 'certificate_type']);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('activity_certificates');
    }
};
