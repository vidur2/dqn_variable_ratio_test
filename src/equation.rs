
#[macro_export]
macro_rules! equation {
    ( $($expr:expr) *, $($value:literal), * ) => {
        let mut answer = 0;
        $(
            answer += $expr * $value;
        )*
        answer
    };

}

fn main(){
    equation!(-5{} + 7{}, 6, 7, 8);
}