package local.guanye;
import java.util.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.aopalliance.intercept.MethodInterceptor;
import org.springframework.aop.framework.ProxyFactory;
import tools.jackson.databind.ObjectMapper;

class ApiProxyTest {
 static class Database extends WorkflowTest.StubJdbc {
  int updates=0;
  @Override public int update(String sql,Object... args){updates++;return 1;}
 }
 Workflow proxy(Database db){
  var factory=new ProxyFactory(new Workflow(db,new ObjectMapper()));
  factory.setProxyTargetClass(true);
  factory.addAdvice((MethodInterceptor)invocation->invocation.proceed());
  return (Workflow)factory.getProxy();
 }
 @Test void projectListingWorksWhenWorkflowIsTransactionProxied(){
  var db=new Database();var user=new WorkflowTest().user();
  var expected=List.<Map<String,Object>>of(Map.of("name","proxy regression"));
  db.answer("SELECT p.*,m.role FROM projects p JOIN memberships m ON m.project_id=p.id WHERE m.user_id=? ORDER BY p.created DESC",expected,user.getSubject());
  var api=new Api(proxy(db),db,new ObjectMapper());
  assertEquals(expected,api.projects(user));
 }
 @Test void deviceHeartbeatUsesInjectedDatabaseWithProxiedWorkflow(){
  var db=new Database();var id=UUID.randomUUID();
  db.answer("SELECT * FROM devices WHERE id=?",List.of(Map.of("id",id,"status","approved")),id);
  var jwt=org.springframework.security.oauth2.jwt.Jwt.withTokenValue("fixture").header("alg","HS256").subject(id.toString()).build();
  var api=new DeviceApi(proxy(db),db);
  assertEquals(Map.of("ok",true),api.heartbeat(jwt));assertEquals(1,db.updates);
 }
}
