package local.guanye;
import java.time.Instant;
import java.util.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.web.server.ResponseStatusException;
import tools.jackson.databind.ObjectMapper;
class WorkflowTest {
 static class StubJdbc extends JdbcTemplate {
  record Query(String sql,List<Object> values){}
  final Map<Query,List<Map<String,Object>>> answers=new HashMap<>();
  void answer(String sql,List<Map<String,Object>> rows,Object... values){answers.put(new Query(sql,Arrays.asList(values)),rows);}
  @Override public List<Map<String,Object>> queryForList(String sql,Object... values){var q=new Query(sql,Arrays.asList(values));if(!answers.containsKey(q))throw new AssertionError("Unexpected database query: "+sql);return answers.get(q);}
 }
 final StubJdbc db=new StubJdbc();final Workflow w=new Workflow(db,new ObjectMapper());
 Jwt user(){return Jwt.withTokenValue("test-only").header("alg","HS256").subject("test-owner").issuedAt(Instant.now()).expiresAt(Instant.now().plusSeconds(60)).build();}
 @Test void unknownProjectIsNotDisclosed(){UUID id=UUID.randomUUID();db.answer("SELECT role FROM memberships WHERE project_id=? AND user_id=?",List.of(),id,"test-owner");assertEquals(404,assertThrows(ResponseStatusException.class,()->w.role(id,user())).getStatusCode().value());}
 @Test void collectorCannotApproveDevices(){UUID p=UUID.randomUUID();UUID d=UUID.randomUUID();db.answer("SELECT * FROM devices WHERE id=?",List.of(Map.of("project_id",p)),d);db.answer("SELECT role FROM memberships WHERE project_id=? AND user_id=?",List.of(Map.of("role","COLLECTOR")),p,"test-owner");assertEquals(403,assertThrows(ResponseStatusException.class,()->w.deviceStatus(user(),d,"approved")).getStatusCode().value());}
 @Test void reviewerCannotImportOtherUserObjects(){UUID p=UUID.randomUUID();db.answer("SELECT role FROM memberships WHERE project_id=? AND user_id=?",List.of(Map.of("role","ADMIN")),p,"test-owner");assertEquals(400,assertThrows(ResponseStatusException.class,()->w.sample(user(),p,"source","users/another/photo.jpg","a".repeat(64),Map.of(),"answer")).getStatusCode().value());}
 @Test void confirmedReviewRequiresHumanLabel(){UUID p=UUID.randomUUID();UUID id=UUID.randomUUID();db.answer("SELECT * FROM platform_samples WHERE id=?",List.of(Map.of("project_id",p)),id);db.answer("SELECT role FROM memberships WHERE project_id=? AND user_id=?",List.of(Map.of("role","REVIEWER")),p,"test-owner");assertEquals(400,assertThrows(ResponseStatusException.class,()->w.review(user(),id,"confirmed","","notes")).getStatusCode().value());}
 @Test void rowTableCannotBeInjected(){assertThrows(IllegalArgumentException.class,()->w.row("users; DROP TABLE users",UUID.randomUUID()));}
}
