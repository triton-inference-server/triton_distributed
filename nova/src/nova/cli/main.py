import click

@click.group()
def main():
    """Nova CLI tool"""
    pass

@click.command()
def serve():
    """Start the Nova server"""
    click.echo("Serve command (not implemented)")

main.add_command(serve)

if __name__ == "__main__":
    main()
