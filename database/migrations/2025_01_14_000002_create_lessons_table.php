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
        Schema::create('lessons', function (Blueprint $table) {
            $table->id();
            $table->foreignId('course_id')->constrained()->onDelete('cascade');
            $table->string('title');
            $table->string('slug');
            $table->text('description')->nullable();
            $table->longText('content')->nullable(); // Main lesson content (HTML/Markdown)
            $table->integer('order_index')->default(0);
            $table->integer('duration_minutes')->default(0);
            $table->integer('is_locked')->default(0);
            $table->integer('is_published')->default(1);
            $table->string('lesson_type')->default('text'); // text, video, quiz, interactive
            $table->json('objectives')->nullable(); // Learning objectives
            $table->json('prerequisites')->nullable(); // Prerequisite lessons
            $table->timestamps();
            
            $table->index(['course_id', 'order_index']);
            $table->index('slug');
            $table->unique(['course_id', 'slug']);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('lessons');
    }
};
