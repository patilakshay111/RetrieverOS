"""Send workspace invitations through an explicitly configured SMTP server."""

from email.message import EmailMessage
import smtplib
import ssl

from rag_pipeline.config import Settings


class SMTPInvitationMailer:
    def __init__(self, settings: Settings):
        host = settings.smtp_host
        sender = settings.smtp_from
        if not host or not sender:
            raise RuntimeError("SMTP invitation delivery is not configured.")
        if bool(settings.smtp_username) != bool(settings.smtp_password):
            raise RuntimeError("SMTP credentials are incomplete.")
        self._host = host
        self._port = settings.smtp_port
        self._sender = sender
        self._starttls = settings.smtp_use_starttls
        self._credentials = (
            (settings.smtp_username, settings.smtp_password)
            if settings.smtp_username and settings.smtp_password
            else None
        )

    def send_invitation(self, email: str, invite_url: str) -> None:
        message = EmailMessage()
        message["Subject"] = "You are invited to a Hybrid RAG workspace"
        message["From"] = self._sender
        message["To"] = email
        message.set_content(
            "You have been invited to join a Hybrid RAG workspace.\n\n"
            f"Accept the invitation: {invite_url}\n"
        )
        try:
            with smtplib.SMTP(
                self._host,
                self._port,
                timeout=10,
            ) as client:
                client.ehlo()
                if self._starttls:
                    client.starttls(context=ssl.create_default_context())
                    client.ehlo()
                if self._credentials is not None:
                    client.login(*self._credentials)
                client.send_message(message)
        except (OSError, smtplib.SMTPException) as error:
            raise RuntimeError("Unable to deliver workspace invitation.") from error
