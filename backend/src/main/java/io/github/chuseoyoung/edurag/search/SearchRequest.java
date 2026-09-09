package io.github.chuseoyoung.edurag.search;

import java.util.Set;
import java.util.UUID;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

public record SearchRequest(
        @NotNull @JsonProperty("client_id") UUID clientId,
        @NotBlank @Size(max = 500) String question,
        @Min(1) @Max(10) @JsonProperty("top_k") Integer topK,
        @JsonProperty("school_level") String schoolLevel,
        @Size(max = 500) @JsonProperty("previous_question") String previousQuestion
) {
    private static final Set<String> SCHOOL_LEVELS = Set.of(
            "all", "elementary", "middle", "high"
    );

    public SearchRequest {
        topK = topK == null ? 5 : topK;
        schoolLevel = schoolLevel == null || schoolLevel.isBlank() ? "all" : schoolLevel;
        if (!SCHOOL_LEVELS.contains(schoolLevel)) {
            throw new IllegalArgumentException("school_level은 all, elementary, middle, high 중 하나여야 합니다.");
        }
    }
}
