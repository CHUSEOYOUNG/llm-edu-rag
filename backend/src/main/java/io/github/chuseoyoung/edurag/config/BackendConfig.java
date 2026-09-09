package io.github.chuseoyoung.edurag.config;

import java.net.http.HttpClient;
import java.util.List;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.web.client.RestClient;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class BackendConfig {

    @Bean
    RestClient aiRestClient(AiServiceProperties properties) {
        var httpClient = HttpClient.newBuilder()
                .connectTimeout(properties.connectTimeout())
                .build();
        var requestFactory = new JdkClientHttpRequestFactory(httpClient);
        requestFactory.setReadTimeout(properties.readTimeout());

        return RestClient.builder()
                .baseUrl(properties.baseUrl())
                .requestFactory(requestFactory)
                .build();
    }

    @Bean
    WebMvcConfigurer corsConfigurer() {
        List<String> localOrigins = List.of(
                "http://127.0.0.1:8765",
                "http://localhost:8765"
        );
        return new WebMvcConfigurer() {
            @Override
            public void addCorsMappings(CorsRegistry registry) {
                registry.addMapping("/api/**")
                        .allowedOrigins(localOrigins.toArray(String[]::new))
                        .allowedMethods("GET", "POST", "DELETE");
            }
        };
    }
}
