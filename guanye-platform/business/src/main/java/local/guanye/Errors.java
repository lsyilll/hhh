package local.guanye;
import java.util.Map;
import org.springframework.web.bind.annotation.*;
import org.springframework.http.HttpStatus;
@RestControllerAdvice class Errors {
 @ExceptionHandler(IllegalArgumentException.class) @ResponseStatus(HttpStatus.BAD_REQUEST) Map<String,String> invalid(IllegalArgumentException e){return Map.of("message","请求参数不正确");}
}
