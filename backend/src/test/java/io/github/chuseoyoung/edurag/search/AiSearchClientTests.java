package io.github.chuseoyoung.edurag.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.content;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.method;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

import java.util.UUID;

import org.junit.jupiter.api.Test;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;

class AiSearchClientTests {

    @Test
    void sendsOnlyTheFastApiSearchFields() {
        RestClient.Builder builder = RestClient.builder()
                .baseUrl("http://127.0.0.1:8765");
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        AiSearchClient client = new AiSearchClient(builder.build());
        var request = new SearchRequest(UUID.randomUUID(), "그럼 중학교는?", 3,
                "middle", "초등학교 수업은 몇 분인가요?");

        server.expect(requestTo("http://127.0.0.1:8765/api/search"))
                .andExpect(method(HttpMethod.POST))
                .andExpect(content().json("""
                        {
                          "question":"그럼 중학교는?",
                          "top_k":3,
                          "school_level":"middle",
                          "previous_question":"초등학교 수업은 몇 분인가요?"
                        }
                        """))
                .andRespond(withSuccess("""
                        {"search_query":"중학교 수업은 몇 분인가요","retrieved_count":2}
                        """, MediaType.APPLICATION_JSON));

        var response = client.search(request);

        assertThat(response.path("retrieved_count").asInt()).isEqualTo(2);
        server.verify();
    }

    @Test
    void sendsAnswerRequestsToTheExplicitGenerationEndpoint() {
        RestClient.Builder builder = RestClient.builder()
                .baseUrl("http://127.0.0.1:8765");
        MockRestServiceServer server = MockRestServiceServer.bindTo(builder).build();
        AiSearchClient client = new AiSearchClient(builder.build());
        var request = new SearchRequest(UUID.randomUUID(), "중학교 수업은 몇 분인가요?",
                2, "middle", null);

        server.expect(requestTo("http://127.0.0.1:8765/api/answer"))
                .andExpect(method(HttpMethod.POST))
                .andRespond(withSuccess("""
                        {"status":"draft_answer","answer":"45분입니다.","search_query":"중학교 수업"}
                        """, MediaType.APPLICATION_JSON));

        var response = client.answer(request);

        assertThat(response.path("status").asString()).isEqualTo("draft_answer");
        server.verify();
    }
}
