package io.github.chuseoyoung.edurag.config;

import java.time.Duration;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties("edu-rag.ai-service")
public record AiServiceProperties(
        String baseUrl,
        Duration connectTimeout,
        Duration readTimeout
) {
}
