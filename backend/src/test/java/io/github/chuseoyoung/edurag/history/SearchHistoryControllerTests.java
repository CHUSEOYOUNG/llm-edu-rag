package io.github.chuseoyoung.edurag.history;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.verify;

import java.util.UUID;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class SearchHistoryControllerTests {

    @Mock
    private SearchHistoryRepository repository;

    @Test
    void deletesOnlyTheRequestedClientsHistoryItem() {
        var historyId = UUID.randomUUID();
        var clientId = UUID.randomUUID();

        var response = new SearchHistoryController(repository).deleteOne(historyId, clientId);

        verify(repository).deleteByIdAndClientId(historyId, clientId);
        assertThat(response.getStatusCode().value()).isEqualTo(204);
    }
}
