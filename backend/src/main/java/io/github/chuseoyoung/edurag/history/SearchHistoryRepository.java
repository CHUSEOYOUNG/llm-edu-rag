package io.github.chuseoyoung.edurag.history;

import java.util.List;
import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;

public interface SearchHistoryRepository extends JpaRepository<SearchHistory, UUID> {

    List<SearchHistory> findTop20ByClientIdOrderByCreatedAtDesc(UUID clientId);

    long deleteByClientId(UUID clientId);
}
