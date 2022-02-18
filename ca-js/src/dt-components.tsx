import { Component } from "react";

export function DtArticle(props: any) {
    console.log(props.children);
    return(
        <div dangerouslySetInnerHTML={{ 
            __html: `
            <dt-article>
                ${props.children}
            </dt-article>
            `}}>
        </div>);
}

export class MyP extends Component {
    text: string;
    constructor(props: {text: string}) {
        super(props);
        this.text = props.text;
    }
    render() {
        return <p>{this.text}</p>
    }
}