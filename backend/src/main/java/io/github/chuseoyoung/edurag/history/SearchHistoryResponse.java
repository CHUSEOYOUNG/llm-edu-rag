package io.github.chuseoyoung.edurag.history;

import java.time.Instant;
import java.util.UUID;

public record SearchHistoryResponse(
        UUID id,
        String question,
        String searchQuery,
        String schoolLevel,
        int resultCount,
        Instant createdAt
) {
    public static SearchHistoryResponse from(SearchHistory history) {
        return new SearchHistoryResponse(
                history.getId(),
                history.getQuestion(),
                history.getSearchQuery(),
                history.getSchoolLevel(),
                history.getResultCount(),
                history.getCreatedAt()
        );
    }
}
