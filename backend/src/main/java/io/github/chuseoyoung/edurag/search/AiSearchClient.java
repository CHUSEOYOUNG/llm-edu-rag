package io.github.chuseoyoung.edurag.search;

import java.util.LinkedHashMap;
import java.util.Map;

import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

@Component
public class AiSearchClient {

    private final RestClient restClient;
    private final ObjectMapper objectMapper;

    public AiSearchClient(RestClient aiRestClient, ObjectMapper objectMapper) {
        this.restClient = aiRestClient;
        this.objectMapper = objectMapper;
    }

    public JsonNode search(SearchRequest request) {
        return post("/api/search", request);
    }

    public JsonNode answer(SearchRequest request) {
        return post("/api/answer", request);
    }

    private JsonNode post(String path, SearchRequest request) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("question", request.question());
        body.put("top_k", request.topK());
        body.put("school_level", request.schoolLevel());
        if (request.previousQuestion() != null && !request.previousQuestion().isBlank()) {
            body.put("previous_question", request.previousQuestion());
        }

        byte[] json = objectMapper.writeValueAsBytes(body);
        return restClient.post()
                .uri(path)
                .contentType(MediaType.APPLICATION_JSON)
                .body(json)
                .retrieve()
                .body(JsonNode.class);
    }
}
