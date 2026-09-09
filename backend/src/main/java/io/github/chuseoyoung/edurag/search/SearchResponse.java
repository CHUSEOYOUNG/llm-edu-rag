package io.github.chuseoyoung.edurag.search;

import java.time.Instant;
import java.util.UUID;

import tools.jackson.databind.JsonNode;

public record SearchResponse(
        @com.fasterxml.jackson.annotation.JsonProperty("history_id") UUID historyId,
        @com.fasterxml.jackson.annotation.JsonProperty("saved_at") Instant savedAt,
        JsonNode search
) {
}
