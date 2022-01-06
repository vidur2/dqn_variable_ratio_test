
#[macro_export]
macro_rules! equation {
    ( $starter: literal $start_var: block $( + $literal: literal $variable: block ) *, $( $value:literal ), * ) => {
        {
            let mut answer = $starter * $value;
            $(
                answer += $literal * $value;
            )*
            answer
        }
    };

}

fn main(){
    println!("{}", equation!(-5{} + 7{}, 6, 7));
}