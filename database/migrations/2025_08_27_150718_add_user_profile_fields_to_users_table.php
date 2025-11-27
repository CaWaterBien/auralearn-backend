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
        Schema::table('users', function (Blueprint $table) {
            $table->string('full_name')->after('name')->nullable();
            $table->string('avatar')->nullable();
            $table->date('join_date')->nullable();
            $table->json('progress')->nullable();
            $table->json('preferences')->nullable();
            $table->boolean('is_active')->default(true);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('users', function (Blueprint $table) {
            $table->dropColumn([
                'full_name', 
                'avatar', 
                'join_date', 
                'progress', 
                'preferences', 
                'is_active'
            ]);
        });
    }
};
