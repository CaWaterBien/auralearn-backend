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
        Schema::create('courses', function (Blueprint $table) {
            $table->id();
            $table->string('title');
            $table->string('slug')->unique();
            $table->text('description')->nullable();
            $table->string('category')->default('web-development'); // web-development, programming, design, etc.
            $table->string('difficulty_level')->default('beginner'); // beginner, intermediate, advanced
            $table->integer('total_lessons')->default(0);
            $table->decimal('duration_hours', 5, 1)->default(0);
            $table->json('tags')->nullable(); // ['HTML', 'CSS', 'JavaScript']
            $table->string('thumbnail')->nullable();
            $table->integer('is_free')->default(1);
            $table->integer('is_published')->default(0);
            $table->integer('order_index')->default(0);
            $table->json('metadata')->nullable(); // Additional course info
            $table->timestamps();
            
            $table->index('slug');
            $table->index('is_published');
            $table->index('order_index');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('courses');
    }
};
