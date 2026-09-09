package io.github.chuseoyoung.edurag;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class EduRagBackendApplication {

	public static void main(String[] args) {
		SpringApplication.run(EduRagBackendApplication.class, args);
	}

}
