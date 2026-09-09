package io.github.chuseoyoung.edurag.search;

import java.util.LinkedHashMap;
import java.util.Map;

import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import tools.jackson.databind.JsonNode;

@Component
public class AiSearchClient {

    private final RestClient restClient;

    public AiSearchClient(RestClient aiRestClient) {
        this.restClient = aiRestClient;
    }

    public JsonNode search(SearchRequest request) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("question", request.question());
        body.put("top_k", request.topK());
        body.put("school_level", request.schoolLevel());
        if (request.previousQuestion() != null && !request.previousQuestion().isBlank()) {
            body.put("previous_question", request.previousQuestion());
        }

        return restClient.post()
                .uri("/api/search")
                .body(body)
                .retrieve()
                .body(JsonNode.class);
    }
}
