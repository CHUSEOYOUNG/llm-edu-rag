package io.github.chuseoyoung.edurag.history;

import java.util.List;
import java.util.UUID;

import org.springframework.http.ResponseEntity;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/search-history")
public class SearchHistoryController {

    private final SearchHistoryRepository repository;

    public SearchHistoryController(SearchHistoryRepository repository) {
        this.repository = repository;
    }

    @GetMapping
    public List<SearchHistoryResponse> list(@RequestParam UUID clientId) {
        return repository.findTop20ByClientIdOrderByCreatedAtDesc(clientId)
                .stream()
                .map(SearchHistoryResponse::from)
                .toList();
    }

    @DeleteMapping
    @Transactional
    public ResponseEntity<Void> delete(@RequestParam UUID clientId) {
        repository.deleteByClientId(clientId);
        return ResponseEntity.noContent().build();
    }

    @DeleteMapping("/{historyId}")
    @Transactional
    public ResponseEntity<Void> deleteOne(@PathVariable UUID historyId,
                                          @RequestParam UUID clientId) {
        repository.deleteByIdAndClientId(historyId, clientId);
        return ResponseEntity.noContent().build();
    }
}
