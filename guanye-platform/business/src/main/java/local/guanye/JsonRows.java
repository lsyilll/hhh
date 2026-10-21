package local.guanye;
import java.util.*;
import org.springframework.core.MethodParameter;
import org.springframework.http.*;
import org.springframework.http.converter.HttpMessageConverter;
import org.springframework.http.server.*;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.servlet.mvc.method.annotation.ResponseBodyAdvice;
import tools.jackson.databind.ObjectMapper;
@RestControllerAdvice class JsonRows implements ResponseBodyAdvice<Object> {
 final ObjectMapper mapper;JsonRows(ObjectMapper mapper){this.mapper=mapper;}
 public boolean supports(MethodParameter type,Class<? extends HttpMessageConverter<?>> converter){return true;}
 Object clean(Object value){if(value==null)return null;if(value.getClass().getName().equals("org.postgresql.util.PGobject"))return mapper.readValue(value.toString(),Object.class);if(value instanceof Map<?,?> map){Map<String,Object> result=new LinkedHashMap<>();map.forEach((k,v)->result.put(k.toString(),clean(v)));return result;}if(value instanceof Collection<?> list)return list.stream().map(this::clean).toList();return value;}
 public Object beforeBodyWrite(Object body,MethodParameter type,MediaType media,Class<? extends HttpMessageConverter<?>> converter,ServerHttpRequest request,ServerHttpResponse response){return clean(body);}
}
