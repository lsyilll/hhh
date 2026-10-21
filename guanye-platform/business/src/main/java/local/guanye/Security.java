package local.guanye;
import javax.crypto.spec.SecretKeySpec;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.oauth2.jose.jws.MacAlgorithm;
import org.springframework.security.oauth2.jwt.*;
@Configuration public class Security {
 @Bean JwtDecoder jwtDecoder(@Value("${guanye.bridge-secret}") String secret){
  if(secret.length()<43)throw new IllegalStateException("Bridge secret must be a generated secret with at least 43 characters");
  var decoder=NimbusJwtDecoder.withSecretKey(new SecretKeySpec(secret.getBytes(java.nio.charset.StandardCharsets.UTF_8),"HmacSHA256")).macAlgorithm(MacAlgorithm.HS256).build();
  decoder.setJwtValidator(new org.springframework.security.oauth2.core.DelegatingOAuth2TokenValidator<>(JwtValidators.createDefaultWithIssuer("guanye-local"),new JwtClaimValidator<java.util.List<String>>("aud",a->a!=null&&a.contains("guanye-business"))));return decoder;
 }
 @Bean SecurityFilterChain filter(HttpSecurity http)throws Exception{return http.csrf(c->c.disable()).sessionManagement(s->s.sessionCreationPolicy(org.springframework.security.config.http.SessionCreationPolicy.STATELESS)).authorizeHttpRequests(a->a.requestMatchers("/actuator/health").permitAll().requestMatchers("/actuator/prometheus").hasAuthority("SCOPE_metrics").requestMatchers("/device/**").hasAuthority("SCOPE_device").anyRequest().hasAuthority("SCOPE_business")).oauth2ResourceServer(o->o.jwt(j->{})).build();}
}
