package io.github.chuseoyoung.edurag.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.UUID;

import io.github.chuseoyoung.edurag.history.SearchHistory;
import io.github.chuseoyoung.edurag.history.SearchHistoryRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import tools.jackson.databind.ObjectMapper;

@ExtendWith(MockitoExtension.class)
class SearchServiceTests {

    @Mock
    private AiSearchClient aiSearchClient;

    @Mock
    private SearchHistoryRepository historyRepository;

    @Test
    void savesQuestionAndResolvedSearchQuery() throws Exception {
        var clientId = UUID.randomUUID();
        var request = new SearchRequest(clientId, "그럼 중학교는?", 3,
                "middle", "초등학교 수업은 몇 분인가요?");
        var result = new ObjectMapper().readTree("""
                {"search_query":"중학교 수업은 몇 분인가요","retrieved_count":2}
                """);
        when(aiSearchClient.search(request)).thenReturn(result);
        when(historyRepository.save(any(SearchHistory.class)))
                .thenAnswer(invocation -> invocation.getArgument(0));

        var response = new SearchService(aiSearchClient, historyRepository).search(request);

        var captor = ArgumentCaptor.forClass(SearchHistory.class);
        verify(historyRepository).save(captor.capture());
        assertThat(captor.getValue().getClientId()).isEqualTo(clientId);
        assertThat(captor.getValue().getSearchQuery()).isEqualTo("중학교 수업은 몇 분인가요");
        assertThat(captor.getValue().getResultCount()).isEqualTo(2);
        assertThat(response.search()).isEqualTo(result);
    }
}
