package io.github.chuseoyoung.edurag.history;

import java.time.Instant;
import java.util.UUID;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "search_history")
public class SearchHistory {

    @Id
    private UUID id;

    @Column(name = "client_id", nullable = false)
    private UUID clientId;

    @Column(nullable = false, length = 500)
    private String question;

    @Column(name = "search_query", nullable = false, length = 1000)
    private String searchQuery;

    @Column(name = "school_level", nullable = false, length = 20)
    private String schoolLevel;

    @Column(name = "result_count", nullable = false)
    private int resultCount;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    protected SearchHistory() {
    }

    public SearchHistory(UUID clientId, String question, String searchQuery,
                         String schoolLevel, int resultCount) {
        this.id = UUID.randomUUID();
        this.clientId = clientId;
        this.question = question;
        this.searchQuery = searchQuery;
        this.schoolLevel = schoolLevel;
        this.resultCount = resultCount;
        this.createdAt = Instant.now();
    }

    public UUID getId() {
        return id;
    }

    public UUID getClientId() {
        return clientId;
    }

    public String getQuestion() {
        return question;
    }

    public String getSearchQuery() {
        return searchQuery;
    }

    public String getSchoolLevel() {
        return schoolLevel;
    }

    public int getResultCount() {
        return resultCount;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }
}
