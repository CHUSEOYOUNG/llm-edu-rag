package io.github.chuseoyoung.edurag.search;

import io.github.chuseoyoung.edurag.history.SearchHistory;
import io.github.chuseoyoung.edurag.history.SearchHistoryRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import tools.jackson.databind.JsonNode;

@Service
public class SearchService {

    private final AiSearchClient aiSearchClient;
    private final SearchHistoryRepository historyRepository;

    public SearchService(AiSearchClient aiSearchClient, SearchHistoryRepository historyRepository) {
        this.aiSearchClient = aiSearchClient;
        this.historyRepository = historyRepository;
    }

    @Transactional
    public SearchResponse search(SearchRequest request) {
        JsonNode result = aiSearchClient.search(request);
        if (result == null) {
            throw new IllegalStateException("AI 검색 서비스가 빈 응답을 반환했습니다.");
        }

        String searchQuery = result.path("search_query").stringValue();
        if (searchQuery == null || searchQuery.isBlank()) {
            searchQuery = request.question();
        }
        int resultCount = result.path("retrieved_count").asInt(0);
        SearchHistory saved = historyRepository.save(new SearchHistory(
                request.clientId(),
                request.question(),
                searchQuery,
                request.schoolLevel(),
                resultCount
        ));

        return new SearchResponse(saved.getId(), saved.getCreatedAt(), result);
    }
}
